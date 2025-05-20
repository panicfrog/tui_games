mod env;
mod games;
mod q_learning;
use std::{thread};
use std::sync::{mpsc, Arc, Mutex};
use std::{error::Error, io, time::Duration};
use env::Env;

use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use crossterm::{execute, terminal::Clear, terminal::ClearType};
use games::maze::map::Action;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    widgets::{Block, Borders, Paragraph},
};

use games::maze::game::{Game, InputAction};

fn keycode_to_action(key: KeyCode, in_victory: bool) -> Option<InputAction> {
    use InputAction::*;
    use crossterm::event::KeyCode::*;

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
    let mut game = Game::new(41, 21);
    let (tx, rx) = mpsc::channel();
    if replaying {
        let max_steps = game.maze.max_steps();
        let q = q_learning::q_learning(&mut game, 999, max_steps, 0.1, 0.9, 0.1);
        let actions = q_learning::replay_best_path(&mut game, &q, max_steps);
        game.reset();
        let actions_clone = actions.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(1));
            for action in actions_clone {
                tx.send(Some(action)).unwrap();
                thread::sleep(Duration::from_millis(150));
            }
            tx.send(None).unwrap();
        });
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
                                execute!(
                                    std::io::stdout(),
                                    Clear(ClearType::All)
                                )?;
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
