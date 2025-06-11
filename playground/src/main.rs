use candle_core::{Device, Result as CandleResult, Tensor};
use env::Env;
use games::maze::map::Action;
use machine_learning::q_learning::{self, rayon_parallel_q_learning};
use machine_learning::q_learning::{
    deep_q_learning, replay_best_path_dqn, ActionDecoder, DQNAgent, Experience, StateEncoder,
};
use std::sync::mpsc;
use std::thread;
use std::{error::Error, io, time::Duration};

use games::crossterm::event::{self, Event, KeyCode};
use games::crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use games::crossterm::{execute, terminal::Clear, terminal::ClearType};
use games::ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};

use games::maze::game::{Game, InputAction};
use games::maze::map::Position;

struct MazeStateEncoder {
    width: usize,
    height: usize,
}

impl MazeStateEncoder {
    fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}

impl StateEncoder<Game> for MazeStateEncoder {
    fn encode(&self, state: &Position) -> CandleResult<Tensor> {
        let mut state_vec = Vec::new();

        // 简化版本：只使用位置one-hot编码和基本距离信息
        let position_index = state.y * self.width + state.x;
        let mut position_one_hot = vec![0.0f32; self.width * self.height];
        if position_index < position_one_hot.len() {
            position_one_hot[position_index] = 1.0;
        }
        state_vec.extend_from_slice(&position_one_hot);

        // 只添加归一化坐标
        let norm_x = (state.x as f32) / (self.width as f32 - 1.0);
        let norm_y = (state.y as f32) / (self.height as f32 - 1.0);
        state_vec.push(norm_x);
        state_vec.push(norm_y);

        let vec_len = state_vec.len();
        Tensor::from_vec(state_vec, (vec_len,), &Device::Cpu)
    }

    fn state_size(&self) -> usize {
        self.width * self.height + 2 // one-hot + 归一化坐标
    }
}

struct MazeActionDecoder;

impl MazeActionDecoder {
    fn new() -> Self {
        Self
    }
}

impl ActionDecoder<Game> for MazeActionDecoder {
    fn decode(&self, action_idx: usize) -> Action {
        match action_idx {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            3 => Action::Right,
            _ => Action::Up,
        }
    }

    fn encode(&self, action: &Action) -> usize {
        match action {
            Action::Up => 0,
            Action::Down => 1,
            Action::Left => 2,
            Action::Right => 3,
        }
    }

    fn action_size(&self) -> usize {
        4
    }

    fn get_legal_action_mask(&self, env: &Game) -> Vec<bool> {
        let legal_actions = env.legal_actions(None);
        let mut mask = vec![false; 4];

        for action in legal_actions {
            let idx = self.encode(&action);
            mask[idx] = true;
        }

        mask
    }
}

fn train_and_replay_with_dqn(
    game: &mut Game,
    width: usize,
    height: usize,
) -> Result<Option<Vec<Action>>, Box<dyn Error>> {
    println!("开始使用DQN训练...");

    let device = Device::Cpu;
    let state_encoder = MazeStateEncoder::new(width, height);
    let action_decoder = MazeActionDecoder::new();

    // 改进DQN超参数 - 简化版本用于快速测试
    let hidden_size = 64; // 减少网络大小
    let buffer_capacity = 5000; // 减少缓冲区大小
    let learning_rate = 0.001; // 提高学习率
    let gamma = 0.9; // 降低折扣因子
    let epsilon = 1.0;
    let epsilon_decay = 0.99; // 更快的epsilon衰减
    let epsilon_min = 0.1; // 更高的最小epsilon
    let batch_size = 32; // 减少批次大小
    let target_update_freq = 50; // 更频繁的目标网络更新

    let agent = DQNAgent::new(
        state_encoder,
        action_decoder,
        device,
        hidden_size,
        buffer_capacity,
        learning_rate,
        gamma,
        epsilon,
        epsilon_decay,
        epsilon_min,
        batch_size,
        target_update_freq,
    )
    .map_err(|e| format!("创建DQN智能体失败: {:?}", e))?;

    let episodes = 100; // 大幅减少训练轮数用于快速测试
    let max_steps = game.maze.max_steps();

    let mut trained_agent = deep_q_learning(game, agent, episodes, max_steps)
        .map_err(|e| format!("DQN训练失败: {:?}", e))?;

    println!("DQN训练完成，开始生成最佳路径...");

    let actions = replay_best_path_dqn(game, &mut trained_agent, max_steps)
        .map_err(|e| format!("重放路径失败: {:?}", e))?;

    if actions.is_some() {
        println!("DQN成功找到解决方案！");
    } else {
        println!("DQN未能找到解决方案");
    }

    Ok(actions)
}

fn keycode_to_action(key: KeyCode, in_victory: bool) -> Option<InputAction> {
    use games::crossterm::event::KeyCode::*;
    use InputAction::*;

    if in_victory {
        match key {
            Left | Right => Some(ToggleButton),
            Enter => Some(Confirm),
            _ => None,
        }
    } else {
        match key {
            Up => Some(MoveUp),
            Down => Some(MoveDown),
            Left => Some(MoveLeft),
            Right => Some(MoveRight),
            Char('q') => Some(Quit),
            _ => None,
        }
    }
}

fn convert_action(action: Action) -> InputAction {
    use InputAction::*;
    match action {
        Action::Up => MoveUp,
        Action::Down => MoveDown,
        Action::Left => MoveLeft,
        Action::Right => MoveRight,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    let backend = CrosstermBackend::new(&mut stdout);
    let mut terminal = Terminal::new(backend)?;
    execute!(std::io::stdout(), Clear(ClearType::All))?;

    let mut replaying = true;
    let width = 41;
    let height = 21;
    let mut game = Game::new(width, height);
    let (tx, rx) = mpsc::channel();

    if replaying {
        // println!("开始训练...");
        let max_steps = game.maze.max_steps();
        // println!("迷宫最大步数: {}", max_steps);

        // let q = q_learning::q_learning(&mut game, width * height * 4, max_steps, 0.1, 0.9);
        // let cpus = num_cpus::get();
        // println!("cpus: {}", cpus);
        // let q = rayon_parallel_q_learning(
        //     &game,
        //     width * height * 20 * (cpus / 4),
        //     max_steps,
        //     0.1,
        //     0.9,
        //     cpus - 1,
        //     40,
        // );
        // println!("{:?}", &q);
        // let actions = q_learning::replay_best_path(&mut game, &q, max_steps);
        // let (q1, q2) = q_learning::double_q_learning(&mut game, width * height * 20, max_steps, 0.1, 0.9);
        // let actions = q_learning::replay_best_path_double_q(&mut game, &q1, &q2, max_steps);

        // 选择使用的算法：true为DQN，false为传统Q-learning
        let use_dqn = true;

        let actions = if use_dqn {
            // 使用DQN
            match train_and_replay_with_dqn(&mut game, width, height) {
                Ok(actions) => actions,
                Err(e) => {
                    println!("DQN失败，回退到传统Q-learning: {}", e);
                    // 如果DQN失败，回退到传统Q-learning
                    let q =
                        q_learning::q_learning(&mut game, width * height * 4, max_steps, 0.1, 0.9);
                    q_learning::replay_best_path(&mut game, &q, max_steps)
                }
            }
        } else {
            // 使用传统Q-learning
            // println!("使用传统Q-learning训练...");
            let q = q_learning::q_learning(&mut game, width * height * 4, max_steps, 0.1, 0.9);
            q_learning::replay_best_path(&mut game, &q, max_steps)
        };

        // 重置游戏状态
        game.reset();

        if let Some(actions) = actions {
            // println!("找到解决方案，路径长度: {}", actions.len());
            let actions_clone = actions.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(300));
                for action in actions_clone {
                    tx.send(Some(action)).unwrap();
                    thread::sleep(Duration::from_millis(150));
                }
                tx.send(None).unwrap();
            });
        } else {
            // println!("未找到解决方案");
            tx.send(None).unwrap();
        }
    }

    loop {
        terminal.draw(|f| {
            let layout = Layout::default()
                .constraints([Constraint::Min(0)])
                .split(f.area());
            let (content, title) = game.render();
            let para =
                Paragraph::new(content).block(Block::default().borders(Borders::ALL).title(title));
            f.render_widget(para, layout[0]);
        })?;

        if replaying {
            if let Ok(msg) = rx.try_recv() {
                if let Some(action) = msg {
                    game.handle_action(convert_action(action));
                } else {
                    replaying = false;
                }
            }
        } else {
            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == event::KeyEventKind::Press {
                        if let Some(action) = keycode_to_action(key.code, game.finished) {
                            if game.handle_action(action) {
                                execute!(std::io::stdout(), Clear(ClearType::All))?;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    disable_raw_mode()?;
    Ok(())
}
