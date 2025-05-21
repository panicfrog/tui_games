mod q_learning;
mod q_utils;

pub use q_learning::{double_q_learning, q_learning, replay_best_path, replay_best_path_double_q};
