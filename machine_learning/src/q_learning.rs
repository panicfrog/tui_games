mod deep_q_learning;
mod parallel_q_learning;
mod q_learning;
mod q_utils;

pub use deep_q_learning::{
    deep_q_learning, replay_best_path_dqn, ActionDecoder, DQNAgent, Experience, ReplayBuffer,
    StateEncoder, DQN,
};
pub use parallel_q_learning::rayon_parallel_q_learning;
pub use q_learning::{double_q_learning, q_learning, replay_best_path, replay_best_path_double_q};
