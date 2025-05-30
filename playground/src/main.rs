use env::Env;
use games::maze::map::Action;
use machine_learning::q_learning::{self, rayon_parallel_q_learning};
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
// 
    let mut replaying = true;
    let width = 41;
    let height = 21;
    let mut game = Game::new(width, height);
    let (tx, rx) = mpsc::channel();
    if replaying {
        let max_steps = game.maze.max_steps();
        let q = q_learning::q_learning(&mut game, width * height * 4, max_steps, 0.1, 0.9);
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
        let actions = q_learning::replay_best_path(&mut game, &q, max_steps);
        // let (q1, q2) = q_learning::double_q_learning(&mut game, width * height * 20, max_steps, 0.1, 0.9);
        // let actions = q_learning::replay_best_path_double_q(&mut game, &q1, &q2, max_steps);
        game.reset();
        if let Some(actions) = actions {
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
