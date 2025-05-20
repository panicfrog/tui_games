use crate::env::Env;
use rand::Rng;
use std::collections::HashMap;

pub fn q_learning<E: Env>(
    env: &mut E,
    episodes: usize,
    max_steps: usize,
    alpha: f32,
    gamma: f32
) -> HashMap<(E::State, E::Action), f32>
where
    E::State: Copy + Eq + std::hash::Hash + std::fmt::Debug,
    E::Action: Copy + Eq + std::hash::Hash + std::fmt::Debug,
{
    let mut q_table: HashMap<(E::State, E::Action), f32> = HashMap::new();
    let mut rng = rand::rng();

    let mut win_count = 0;

    for episode in 0..episodes {
        let mut state = env.reset();
        let epsilon = (1.0 - episode as f32 / episodes as f32).max(0.05);
        for _ in 0..max_steps {
            let actions = env.legal_actions(None);
            if actions.is_empty() {
                break;
            }

            // Epsilon-greedy 策略选择动作（过滤掉 NaN）
            let action = if actions.len() == 1 {
                actions[0]
            } else if rng.random::<f32>() < epsilon {
                actions[rng.random_range(0..actions.len())]
            } else {
                match actions
                    .iter()
                    .copied()
                    .filter(|&a| !q_table.get(&(state, a)).unwrap_or(&0.0).is_nan())
                    .max_by(|&a1, &a2| {
                        let q1 = *q_table.get(&(state, a1)).unwrap_or(&0.0);
                        let q2 = *q_table.get(&(state, a2)).unwrap_or(&0.0);
                        q1.partial_cmp(&q2).unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                    Some(a) => a,
                    None => break,
                }
            };

            let (next_state, _status) = env.step(action);

            // 奖励函数
            let reward = if env.is_win() { 10.0 } else if !env.is_terminal() { -0.01 } else { -1.0 };

            // 计算下一个状态的最大 Q 值
            let max_q = env
                .legal_actions(Some((next_state, action)))
                .iter()
                .map(|&a| *q_table.get(&(next_state, a)).unwrap_or(&0.0))
                .filter(|q| !q.is_nan())
                .fold(f32::NEG_INFINITY, f32::max);

            // 更新 Q 表
            let entry = q_table.entry((state, action)).or_insert(0.0);
            // Q(s, a) ← Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]
            *entry += alpha * (reward + gamma * max_q - *entry);

            state = next_state;

            if env.is_win() {
                win_count += 1;
                break;
            }
            if env.is_terminal() {
                break;
            }
        }
    }
    println!("Win rate: {:.2}%", (win_count as f32 / episodes as f32) * 100.0);
    q_table
}

pub fn replay_best_path<E: Env>(
    env: &mut E,
    q_table: &HashMap<(E::State, E::Action), f32>,
    max_steps: usize,
) -> Option<Vec<E::Action>>
where
    E::State: Copy + Eq + std::hash::Hash,
    E::Action: Copy + Eq + std::hash::Hash,
{
    let mut path = vec![];
    let mut state = env.reset();

    for _ in 0..max_steps {
        let actions = env.legal_actions(None);

        let action = if actions.len() == 1 {
            actions[0]
        } else {
            match actions
                .iter()
                .copied()
                .filter(|&a| {
                    let q = *q_table.get(&(state, a)).unwrap_or(&0.0);
                    !q.is_nan()
                })
                .max_by(|a1, a2| {
                    let q1 = *q_table.get(&(state, *a1)).unwrap_or(&0.0);
                    let q2 = *q_table.get(&(state, *a2)).unwrap_or(&0.0);
                    q1.partial_cmp(&q2).unwrap_or(std::cmp::Ordering::Equal)
                }) {
                Some(a) => a,
                None => break,
            }
        };

        let (next_state, _status) = env.step(action);
        path.push(action);
        state = next_state;

        if env.is_win() {
            return Some(path);
        }
        if env.is_terminal() {
            break;
        }
    }

    // 没有成功达到终点
    None
}
