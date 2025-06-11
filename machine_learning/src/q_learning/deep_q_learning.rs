use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{loss, AdamW, Linear, Optimizer, VarBuilder, VarMap};
use env::{rand::Rng, Env};
use std::collections::VecDeque;

// 经验回放缓冲区
#[derive(Clone)]
pub struct Experience<S, A> {
    pub state: S,      // 当前状态
    pub action: A,     // 执行的动作
    pub reward: f32,   // 获得的奖励
    pub next_state: S, // 下一个状态
    pub done: bool,    // 是否为终止状态
}

pub struct ReplayBuffer<S, A> {
    buffer: VecDeque<Experience<S, A>>,
    capacity: usize,
}

impl<S: Clone, A: Clone> ReplayBuffer<S, A> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, experience: Experience<S, A>) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn sample(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<Experience<S, A>> {
        let mut samples = Vec::with_capacity(batch_size);
        let len = self.buffer.len();

        for _ in 0..batch_size.min(len) {
            let idx = rng.random_range(0..len);
            samples.push(self.buffer[idx].clone());
        }

        samples
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

// 深度Q网络
pub struct DQN {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl DQN {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let fc1 = candle_nn::linear(input_size, hidden_size, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden_size, hidden_size, vb.pp("fc2"))?;
        let fc3 = candle_nn::linear(hidden_size, output_size, vb.pp("fc3"))?;

        Ok(Self { fc1, fc2, fc3 })
    }
}

impl Module for DQN {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;
        let x = x.relu()?;
        let x = self.fc3.forward(&x)?;
        Ok(x)
    }
}

// 状态编码器 trait - 将环境状态转换为张量
pub trait StateEncoder<E: Env> {
    fn encode(&self, state: &E::State) -> Result<Tensor>;
    fn state_size(&self) -> usize;
}

// 动作解码器 trait - 在动作索引和环境动作之间转换
pub trait ActionDecoder<E: Env> {
    fn decode(&self, action_idx: usize) -> E::Action;
    fn encode(&self, action: &E::Action) -> usize;
    fn action_size(&self) -> usize;
    fn get_legal_action_mask(&self, env: &E) -> Vec<bool>;
}

// 深度Q学习智能体
pub struct DQNAgent<E: Env, SE: StateEncoder<E>, AD: ActionDecoder<E>> {
    q_network: DQN,
    target_network: DQN,
    optimizer: AdamW,
    replay_buffer: ReplayBuffer<E::State, E::Action>,
    state_encoder: SE,
    action_decoder: AD,
    device: Device,

    // 分离的网络变量映射
    main_varmap: VarMap,
    target_varmap: VarMap,

    // 超参数
    gamma: f32,
    epsilon: f32,
    epsilon_decay: f32,
    epsilon_min: f32,
    learning_rate: f64,
    batch_size: usize,
    target_update_freq: usize,
    step_count: usize,
}

impl<E: Env, SE: StateEncoder<E>, AD: ActionDecoder<E>> DQNAgent<E, SE, AD>
where
    E::State: Clone,
    E::Action: Clone + Copy + std::fmt::Debug,
{
    pub fn new(
        state_encoder: SE,
        action_decoder: AD,
        device: Device,
        hidden_size: usize,
        buffer_capacity: usize,
        learning_rate: f64,
        gamma: f32,
        epsilon: f32,
        epsilon_decay: f32,
        epsilon_min: f32,
        batch_size: usize,
        target_update_freq: usize,
    ) -> Result<Self> {
        let state_size = state_encoder.state_size();
        let action_size = action_decoder.action_size();

        // 分离主网络和目标网络的VarMap
        let main_varmap = VarMap::new();
        let target_varmap = VarMap::new();

        let main_vb = VarBuilder::from_varmap(&main_varmap, DType::F32, &device);
        let target_vb = VarBuilder::from_varmap(&target_varmap, DType::F32, &device);

        // 创建主网络
        let q_network = DQN::new(state_size, hidden_size, action_size, main_vb)?;

        // 创建目标网络（使用独立的VarMap）
        let target_network = DQN::new(state_size, hidden_size, action_size, target_vb)?;

        // 只把主网络参数交给优化器
        let optimizer = AdamW::new(
            main_varmap.all_vars(),
            candle_nn::ParamsAdamW {
                lr: learning_rate,
                ..Default::default()
            },
        )?;

        let replay_buffer = ReplayBuffer::new(buffer_capacity);

        let mut agent = Self {
            q_network,
            target_network,
            optimizer,
            replay_buffer,
            state_encoder,
            action_decoder,
            device,
            main_varmap,
            target_varmap,
            gamma,
            epsilon,
            epsilon_decay,
            epsilon_min,
            learning_rate,
            batch_size,
            target_update_freq,
            step_count: 0,
        };

        // 初始化目标网络权重
        agent.initialize_target_network()?;

        Ok(agent)
    }

    pub fn select_action(
        &mut self,
        env: &E,
        state: &E::State,
        rng: &mut impl Rng,
    ) -> Result<E::Action> {
        // 获取合法动作掩码
        let legal_mask = self.action_decoder.get_legal_action_mask(env);
        let legal_actions: Vec<usize> = legal_mask
            .iter()
            .enumerate()
            .filter(|(_, &is_legal)| is_legal)
            .map(|(idx, _)| idx)
            .collect();

        if legal_actions.is_empty() {
            panic!("No legal actions available");
        }

        // ε-贪婪动作选择
        if rng.random::<f32>() < self.epsilon {
            // 从合法动作中随机选择
            let random_idx = legal_actions[rng.random_range(0..legal_actions.len())];
            Ok(self.action_decoder.decode(random_idx))
        } else {
            // 贪婪动作选择
            let state_tensor = self.state_encoder.encode(state)?;
            let state_batch = state_tensor.unsqueeze(0)?; // 添加批次维度

            let q_values = self.q_network.forward(&state_batch)?;
            let q_values = q_values.squeeze(0)?; // 移除批次维度
            let q_values_vec = q_values.to_vec1::<f32>()?;

            // 使用更稳定的掩码方法
            let best_action_idx = self.select_best_legal_action(&q_values_vec, &legal_mask)?;

            Ok(self.action_decoder.decode(best_action_idx))
        }
    }

    pub fn store_experience(&mut self, experience: Experience<E::State, E::Action>) {
        self.replay_buffer.push(experience);
    }

    pub fn train(&mut self, rng: &mut impl Rng) -> Result<f32> {
        if self.replay_buffer.len() < self.batch_size {
            return Ok(0.0);
        }

        // 从经验回放缓冲区采样批次
        let batch = self.replay_buffer.sample(self.batch_size, rng);

        // 准备批次张量
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut rewards = Vec::new();
        let mut next_states = Vec::new();
        let mut dones = Vec::new();

        for exp in &batch {
            states.push(self.state_encoder.encode(&exp.state)?);
            actions.push(self.action_decoder.encode(&exp.action));
            rewards.push(exp.reward);
            next_states.push(self.state_encoder.encode(&exp.next_state)?);
            dones.push(if exp.done { 1.0f32 } else { 0.0f32 });
        }

        // 堆叠张量
        let state_batch = Tensor::stack(&states, 0)?;
        let next_state_batch = Tensor::stack(&next_states, 0)?;
        let rewards_tensor = Tensor::from_vec(rewards, (self.batch_size,), &self.device)?;
        let dones_tensor = Tensor::from_vec(dones, (self.batch_size,), &self.device)?;

        // 当前Q值
        let current_q_values = self.q_network.forward(&state_batch)?;

        // 选择执行的动作对应的Q值
        let action_indices = Tensor::from_vec(
            actions.iter().map(|&a| a as i64).collect::<Vec<_>>(),
            (self.batch_size,),
            &self.device,
        )?;
        let current_q = current_q_values
            .gather(&action_indices.unsqueeze(1)?, 1)?
            .squeeze(1)?;

        // 使用目标网络计算目标Q值，并截断梯度
        let next_q_values = self.target_network.forward(&next_state_batch)?;
        let max_next_q = next_q_values.max(1)?; // 获取最大值
        let max_next_q = max_next_q.detach(); // 关键：截断梯度

        // 计算目标值：reward + gamma * max_next_q * (1 - done)
        let ones = Tensor::ones((self.batch_size,), DType::F32, &self.device)?;
        let not_done = ones.sub(&dones_tensor)?;
        let gamma_tensor = Tensor::from_vec(
            vec![self.gamma; self.batch_size],
            (self.batch_size,),
            &self.device,
        )?;
        let discounted_next_q = max_next_q.mul(&gamma_tensor)?;
        let target_q = rewards_tensor.add(&discounted_next_q.mul(&not_done)?)?;

        // 计算损失 (MSE)
        let loss = loss::mse(&current_q, &target_q)?;

        // 反向传播
        self.optimizer.backward_step(&loss)?;

        // 更新目标网络（每target_update_freq步）
        self.step_count += 1;
        if self.step_count % self.target_update_freq == 0 {
            self.update_target_network()?;
        }

        Ok(loss.to_scalar::<f32>()?)
    }

    /// 更新epsilon - 移到episode级别调用
    pub fn update_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    /// 更新目标网络权重
    ///
    /// 在DQN算法中，目标网络用于计算目标Q值，以提高训练稳定性。
    /// 目标网络的权重定期从主网络复制，而不是每次都更新。
    fn update_target_network(&mut self) -> Result<()> {
        self.hard_update_target_network()
    }

    /// 按名字复制VarMap，避免顺序错位导致的shape mismatch
    fn copy_varmap(src: &VarMap, dst: &VarMap) -> Result<()> {
        let src_data = src.data().lock().unwrap();
        let dst_data = dst.data().lock().unwrap();

        for (name, src_var) in src_data.iter() {
            let Some(dst_var) = dst_data.get(name) else {
                return Err(candle_core::Error::Msg(format!("目标网络缺少变量 {name}")));
            };
            dst_var.set(&src_var.as_tensor().clone())?; // clone 防止共用存储
        }
        Ok(())
    }

    /// 初始化目标网络权重 - 首次设置时使用
    ///
    /// 将主网络的权重复制到目标网络
    fn initialize_target_network(&mut self) -> Result<()> {
        Self::copy_varmap(&self.main_varmap, &self.target_varmap)
    }

    /// 硬更新：直接复制主网络权重到目标网络
    ///
    /// # Returns
    /// * `Ok(())` - 更新成功
    /// * `Err(Error)` - 网络参数不匹配或其他错误
    fn hard_update_target_network(&mut self) -> Result<()> {
        Self::copy_varmap(&self.main_varmap, &self.target_varmap)
    }

    /// 软更新：目标网络 = τ * 主网络 + (1-τ) * 目标网络
    ///
    /// 软更新是DDPG等算法中常用的方式，每次训练都进行小幅度的权重混合。
    /// 这种方式能提供更平滑的目标值变化，但更新频率更高。
    ///
    /// # Arguments
    /// * `tau` - 软更新系数，通常是一个很小的值（如0.001-0.01）
    ///   - tau=1.0 相当于硬更新
    ///   - tau=0.0 目标网络不变
    ///   - tau越小，目标网络变化越慢
    ///
    /// # Returns
    /// * `Ok(())` - 更新成功
    /// * `Err(Error)` - 网络参数不匹配或计算错误
    #[allow(dead_code)]
    fn soft_update_target_network(&mut self, tau: f32) -> Result<()> {
        let main_vars = self.main_varmap.all_vars();
        let target_vars = self.target_varmap.all_vars();

        if main_vars.len() != target_vars.len() {
            return Err(candle_core::Error::Msg("网络参数数量不匹配".to_string()));
        }

        for (main_var, target_var) in main_vars.iter().zip(target_vars.iter()) {
            let main_data = main_var.as_tensor();
            let target_data = target_var.as_tensor();

            // 软更新公式：target = tau * main + (1 - tau) * target
            let tau_tensor = Tensor::from_vec(vec![tau], (), &self.device)?;
            let one_minus_tau = Tensor::from_vec(vec![1.0 - tau], (), &self.device)?;

            let updated_target = main_data
                .mul(&tau_tensor)?
                .add(&target_data.mul(&one_minus_tau)?)?;
            target_var.set(&updated_target)?;
        }

        Ok(())
    }

    pub fn get_epsilon(&self) -> f32 {
        self.epsilon
    }

    /// 从合法动作中选择Q值最大的动作
    ///
    /// 这个方法避免使用无穷大值，提供更稳定的数值计算。
    /// 相比直接使用 f32::NEG_INFINITY 掩码，这种方法有以下优势：
    ///
    /// 1. **数值稳定性**: 避免无穷大值导致的梯度爆炸或NaN
    /// 2. **优化器友好**: 大多数优化器能更好地处理有限值
    /// 3. **收敛性**: 减少训练过程中的数值异常
    /// 4. **可解释性**: 只考虑合法动作，逻辑更清晰
    ///
    /// # Arguments
    /// * `q_values` - 所有动作的Q值向量
    /// * `legal_mask` - 合法动作掩码，true表示合法动作
    ///
    /// # Returns
    /// * `Ok(usize)` - 最优动作的索引
    /// * `Err(Error)` - 没有合法动作时返回错误
    ///
    /// # Example
    /// ```rust,ignore
    /// let q_values = vec![0.5, -0.2, 0.8, 0.1];
    /// let legal_mask = vec![true, false, true, true];
    /// // 只会从索引 0, 2, 3 中选择，结果是索引 2 (Q值最大的合法动作)
    /// let best_action = agent.select_best_legal_action(&q_values, &legal_mask)?;
    /// ```
    fn select_best_legal_action(&self, q_values: &[f32], legal_mask: &[bool]) -> Result<usize> {
        // 方法1: 只考虑合法动作的Q值
        let legal_q_values: Vec<(usize, f32)> = q_values
            .iter()
            .enumerate()
            .filter(|(idx, _)| legal_mask[*idx])
            .map(|(idx, &q)| (idx, q))
            .collect();

        if legal_q_values.is_empty() {
            return Err(candle_core::Error::Msg("没有可用的合法动作".to_string()));
        }

        // 找到Q值最大的合法动作
        let best_action = legal_q_values
            .iter()
            .max_by(|(_, q1), (_, q2)| q1.partial_cmp(q2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| *idx)
            .unwrap();

        Ok(best_action)
    }

    /// 使用软掩码的动作选择方法 (备选方案)
    ///
    /// 对非法动作使用很小的负值而不是无穷大，提供更好的数值稳定性
    #[allow(dead_code)]
    fn select_best_action_soft_mask(&self, q_values: &[f32], legal_mask: &[bool]) -> Result<usize> {
        let mut masked_q_values = q_values.to_vec();

        // 找到当前Q值的最小值，用于计算惩罚
        let min_q = q_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let penalty = min_q - 1000.0; // 使用大的负值但不是无穷大

        // 对非法动作施加惩罚
        for (idx, &is_legal) in legal_mask.iter().enumerate() {
            if !is_legal {
                masked_q_values[idx] = penalty;
            }
        }

        // 找到最大Q值的动作
        let best_action_idx = masked_q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| candle_core::Error::Msg("无法选择动作".to_string()))?;

        Ok(best_action_idx)
    }

    /// 手动执行目标网络的硬更新
    ///
    /// 这个方法允许用户在训练过程之外手动更新目标网络，
    /// 对于自定义训练循环或调试很有用。
    ///
    /// # Returns
    /// * `Ok(())` - 更新成功
    /// * `Err(Error)` - 更新失败
    ///
    /// # Example
    /// ```rust,ignore
    /// agent.update_target_network_hard()?;
    /// ```
    pub fn update_target_network_hard(&mut self) -> Result<()> {
        self.hard_update_target_network()
    }

    /// 手动执行目标网络的软更新
    ///
    /// 允许用户指定自定义的tau值进行软更新。
    /// 软更新在某些算法（如DDPG、TD3）中很常见。
    ///
    /// # Arguments
    /// * `tau` - 软更新系数，范围通常是[0.001, 0.1]
    ///
    /// # Returns
    /// * `Ok(())` - 更新成功
    /// * `Err(Error)` - 更新失败
    ///
    /// # Example
    /// ```rust,ignore
    /// // 使用0.005的tau值进行软更新
    /// agent.update_target_network_soft(0.005)?;
    /// ```
    pub fn update_target_network_soft(&mut self, tau: f32) -> Result<()> {
        self.soft_update_target_network(tau)
    }
}

// 训练函数
pub fn deep_q_learning<E, SE, AD>(
    env: &mut E,
    mut agent: DQNAgent<E, SE, AD>,
    episodes: usize,
    max_steps: usize,
) -> Result<DQNAgent<E, SE, AD>>
where
    E: Env,
    E::State: Clone,
    E::Action: Clone + Copy + std::fmt::Debug,
    SE: StateEncoder<E>,
    AD: ActionDecoder<E>,
{
    let mut rng = env::rand::rng();
    let mut win_count = 0;
    let mut total_losses = Vec::new();

    for episode in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut episode_loss = 0.0;
        let mut steps = 0;

        for _step in 0..max_steps {
            // 选择动作
            let action = agent.select_action(env, &state, &mut rng)?;

            // 执行动作
            let (next_state, _status) = env.step(action);

            // 改进奖励计算 - 使用更合理的奖励结构
            let reward = if env.is_win() {
                10.0 // 降低胜利奖励，与步数同量级
            } else if env.is_terminal() {
                -1.0 // 适度的失败惩罚
            } else {
                -0.01 // 轻微的步数惩罚，鼓励快速完成
            };

            let done = env.is_terminal() || env.is_win();

            // 存储经验
            agent.store_experience(Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });

            // 训练智能体
            let loss = agent.train(&mut rng)?;
            episode_loss += loss;

            episode_reward += reward;
            state = next_state;
            steps += 1;

            if done {
                if env.is_win() {
                    win_count += 1;
                }
                break;
            }
        }

        // 每个episode结束后才更新epsilon
        agent.update_epsilon();

        total_losses.push(episode_loss / steps as f32);

        // 打印进度
        if episode % 10 == 0 || episode < 10 {
            // 前10个episode和之后每10个episode都输出
            let win_rate = (win_count as f32 / (episode + 1) as f32) * 100.0;
            let avg_loss = if total_losses.is_empty() {
                0.0
            } else {
                total_losses.iter().sum::<f32>() / total_losses.len() as f32
            };
            println!(
                "Episode {}: Win Rate: {:.2}%, Epsilon: {:.3}, Avg Loss: {:.6}, Episode Reward: {:.2}, Steps: {}",
                episode, win_rate, agent.get_epsilon(), avg_loss, episode_reward, steps
            );
        }

        // 每50个episode输出一次详细信息
        if episode % 50 == 0 && episode > 0 {
            println!(
                "  缓冲区大小: {}, 最近10个episode平均奖励: {:.2}",
                agent.replay_buffer.len(),
                if total_losses.len() >= 10 {
                    total_losses.iter().rev().take(10).sum::<f32>() / 10.0
                } else {
                    total_losses.iter().sum::<f32>() / total_losses.len() as f32
                }
            );
        }
    }

    let final_win_rate = (win_count as f32 / episodes as f32) * 100.0;
    println!("训练完成。最终胜率: {:.2}%", final_win_rate);

    Ok(agent)
}

// 使用训练好的DQN回放最佳路径
pub fn replay_best_path_dqn<E, SE, AD>(
    env: &mut E,
    agent: &mut DQNAgent<E, SE, AD>,
    max_steps: usize,
) -> Result<Option<Vec<E::Action>>>
where
    E: Env,
    E::State: Clone,
    E::Action: Clone + Copy + std::fmt::Debug,
    SE: StateEncoder<E>,
    AD: ActionDecoder<E>,
{
    let mut path = Vec::new();
    let mut state = env.reset();
    let mut rng = env::rand::rng();

    // 临时将epsilon设为0以进行贪婪动作选择
    let original_epsilon = agent.epsilon;
    agent.epsilon = 0.0;

    for _ in 0..max_steps {
        let action = agent.select_action(env, &state, &mut rng)?;
        let (next_state, _status) = env.step(action);

        path.push(action);
        state = next_state;

        if env.is_win() {
            agent.epsilon = original_epsilon; // 恢复原始epsilon
            return Ok(Some(path));
        }

        if env.is_terminal() {
            break;
        }
    }

    agent.epsilon = original_epsilon; // 恢复原始epsilon
    Ok(None) // 没有找到成功路径
}
