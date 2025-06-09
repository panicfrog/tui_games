use crate::q_learning::q_utils::random_action;

use super::q_utils::{estimate_double_q_target, estimate_max_q_value};
use env::rand::Rng;
use env::Env;
use std::collections::HashMap;

#[allow(dead_code)]
pub fn q_learning<E: Env>(
    env: &mut E,
    episodes: usize,
    max_steps: usize,
    alpha: f32,
    gamma: f32,
) -> HashMap<(E::State, E::Action), f32>
where
    E::State: Copy + Eq + std::hash::Hash + std::fmt::Debug,
    E::Action: Copy + Eq + std::hash::Hash + std::fmt::Debug,
{
    let mut q_table: HashMap<(E::State, E::Action), f32> = HashMap::new();
    let mut rng = env::rand::rng();
    let mut win_count = 0;

    let win_reward = 10.0;
    let lose_reward = -1.0;
    let step_reward = lose_reward / (max_steps + 1) as f32;

    for episode in 0..episodes {
        let mut state = env.reset();
        let epsilon = (1.0 - episode as f32 / episodes as f32).max(0.05);
        let mut exported_table = HashMap::new();

        let mut forced_actions: Vec<(E::State, E::Action)> = Vec::new();

        for _ in 0..max_steps {
            let actions = env.legal_actions(None);
            if actions.is_empty() {
                break;
            }

            let action = if actions.len() == 1 {
                let action = actions[0];
                forced_actions.push((state, action));
                // println!("{action:?} o");
                action
            } else {
                if !forced_actions.is_empty() {
                    let next_max_q = actions
                        .iter()
                        .map(|&a| *q_table.get(&(state, a)).unwrap_or(&0.0))
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(0.0);

                    let update = alpha
                        * (step_reward + gamma * next_max_q
                            - *q_table.get(&forced_actions[0]).unwrap_or(&0.0));
                    for (s, a) in &forced_actions {
                        *q_table.entry((*s, *a)).or_insert(0.0) += update;
                    }
                    forced_actions.clear();
                }
                if rng.random::<f32>() < epsilon {
                    let action = random_action::<E>(&actions, state, &mut rng, &exported_table);
                    exported_table
                        .entry((state, action))
                        .and_modify(|v| {
                            *v += 1;
                        })
                        .or_insert(0);
                    action
                } else {
                    let a = actions
                        .iter()
                        .copied()
                        .max_by(|&a1, &a2| {
                            let q1 = *q_table.get(&(state, a1)).unwrap_or(&0.0);
                            let q2 = *q_table.get(&(state, a2)).unwrap_or(&0.0);
                            q1.partial_cmp(&q2).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or_else(|| actions[rng.random_range(0..actions.len())]);
                    a
                }
            };

            let (next_state, _status) = env.step(action);

            // 只有在非强制性动作时才立即更新 Q 值
            if actions.len() > 1 {
                let reward = if env.is_win() {
                    win_reward
                } else if !env.is_terminal() {
                    step_reward
                } else {
                    lose_reward
                };

                let target = estimate_max_q_value(env, &q_table, next_state, action);
                let entry = q_table.entry((state, action)).or_insert(0.0);
                *entry += alpha * (reward + gamma * target - *entry);
            }

            state = next_state;

            if env.is_win() || env.is_terminal() {
                if !forced_actions.is_empty() {
                    let final_reward = if env.is_win() { 10.0 } else { -1.0 };

                    let update =
                        alpha * (final_reward - *q_table.get(&forced_actions[0]).unwrap_or(&0.0));
                    for (s, a) in &forced_actions {
                        *q_table.entry((*s, *a)).or_insert(0.0) += update;
                    }
                    forced_actions.clear();
                }

                if env.is_win() {
                    win_count += 1;
                }
                break;
            }
        }
    }
    println!(
        "Win rate: {:.2}%",
        (win_count as f32 / episodes as f32) * 100.0
    );
    q_table
}

#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn double_q_learning<E: Env>(
    env: &mut E,
    episodes: usize,
    max_steps: usize,
    alpha: f32,
    gamma: f32,
) -> (
    HashMap<(E::State, E::Action), f32>,
    HashMap<(E::State, E::Action), f32>,
)
where
    E::State: Copy + Eq + std::hash::Hash + std::fmt::Debug,
    E::Action: Copy + Eq + std::hash::Hash + std::fmt::Debug,
{
    let mut q1: HashMap<(E::State, E::Action), f32> = HashMap::new();
    let mut q2: HashMap<(E::State, E::Action), f32> = HashMap::new();
    let mut rng = env::rand::rng();
    let mut win_count = 0;

    for episode in 0..episodes {
        let mut state = env.reset();
        let epsilon = (1.0 - episode as f32 / episodes as f32).max(0.05);

        for _ in 0..max_steps {
            let actions = env.legal_actions(None);
            if actions.is_empty() {
                break;
            }

            let action = if actions.len() == 1 {
                actions[0]
            } else if rng.random::<f32>() < epsilon {
                actions[rng.random_range(0..actions.len())]
            } else {
                actions
                    .iter()
                    .copied()
                    .max_by(|&a1, &a2| {
                        let q1_sum = q1.get(&(state, a1)).unwrap_or(&0.0)
                            + q2.get(&(state, a1)).unwrap_or(&0.0);
                        let q2_sum = q1.get(&(state, a2)).unwrap_or(&0.0)
                            + q2.get(&(state, a2)).unwrap_or(&0.0);
                        q1_sum
                            .partial_cmp(&q2_sum)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or_else(|| actions[rng.random_range(0..actions.len())])
            };

            let (next_state, _status) = env.step(action);
            let reward = if env.is_win() {
                10.0
            } else if !env.is_terminal() {
                -0.01
            } else {
                -1.0
            };

            if rng.random_bool(0.5) {
                // Update Q1
                let target = estimate_double_q_target(env, &q1, &q2, next_state, action);
                let entry = q1.entry((state, action)).or_insert(0.0);
                *entry += alpha * (reward + gamma * target - *entry);
            } else {
                // Update Q2
                let target = estimate_double_q_target(env, &q2, &q1, next_state, action);
                let entry = q2.entry((state, action)).or_insert(0.0);
                *entry += alpha * (reward + gamma * target - *entry);
            }

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

    println!(
        "Win rate: {:.2}%",
        (win_count as f32 / episodes as f32) * 100.0
    );
    (q1, q2)
}

#[allow(dead_code)]
pub fn replay_best_path_double_q<E: Env>(
    env: &mut E,
    q1: &HashMap<(E::State, E::Action), f32>,
    q2: &HashMap<(E::State, E::Action), f32>,
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
        if actions.is_empty() {
            return None;
        }

        let action = if actions.len() == 1 {
            actions[0]
        } else {
            actions.iter().copied().max_by(|&a1, &a2| {
                let q1_a1 = q1.get(&(state, a1)).unwrap_or(&0.0);
                let q2_a1 = q2.get(&(state, a1)).unwrap_or(&0.0);
                let q_sum1 = q1_a1 + q2_a1;

                let q1_a2 = q1.get(&(state, a2)).unwrap_or(&0.0);
                let q2_a2 = q2.get(&(state, a2)).unwrap_or(&0.0);
                let q_sum2 = q1_a2 + q2_a2;

                q_sum1
                    .partial_cmp(&q_sum2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?
        };

        let (next_state, _status) = env.step(action);
        path.push(action);
        state = next_state;

        if env.is_win() {
            return Some(path);
        }

        if env.is_terminal() {
            return None;
        }
    }

    None
}
