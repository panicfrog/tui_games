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
                        let q1 = *q_table.get(&(state, a1)).unwrap_or(&0.0);
                        let q2 = *q_table.get(&(state, a2)).unwrap_or(&0.0);
                        q1.partial_cmp(&q2).unwrap_or(std::cmp::Ordering::Equal)
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

            let target = estimate_max_q_value(env, &q_table, next_state, action);
            let entry = q_table.entry((state, action)).or_insert(0.0);
            *entry += alpha * (reward + gamma * target - *entry);

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
