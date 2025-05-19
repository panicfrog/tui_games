mod env;
mod games;
mod q_learning;
use std::{error::Error, io, time::Duration};

use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use crossterm::{execute, terminal::Clear, terminal::ClearType};
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

fn main() -> Result<(), Box<dyn Error>> {
    enable_raw_mode()?;
    // let mut stdout = io::stdout();
    // let backend = CrosstermBackend::new(&mut stdout);
    // let mut terminal = Terminal::new(backend)?;
    // execute!(std::io::stdout(), Clear(ClearType::All))?;

    let mut game = Game::new(41, 21);
    let max_steps = game.maze.max_steps();
    let q = q_learning::q_learning(&mut game, 1000, max_steps, 0.1, 0.9, 0.1);
    let actions = q_learning::replay_best_path(&mut game, &q, max_steps);
    println!("Best path: {:?}", actions);
    // loop {
    //     terminal.draw(|f| {
    //         let layout = Layout::default()
    //             .constraints([Constraint::Min(0)])
    //             .split(f.area());
    //         let (content, title) = game.render();
    //         let para =
    //             Paragraph::new(content).block(Block::default().borders(Borders::ALL).title(title));
    //         f.render_widget(para, layout[0]);
    //     })?;

    //     if event::poll(Duration::from_millis(100))? {
    //         if let Event::Key(key) = event::read()? {
    //             if key.kind == event::KeyEventKind::Press {
    //                 if let Some(action) = keycode_to_action(key.code, game.finished) {
    //                     if game.handle_action(action) {
    //                         execute!(
    //                             std::io::stdout(),
    //                             crossterm::terminal::Clear(crossterm::terminal::ClearType::All)
    //                         )?;
    //                         break;
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    // disable_raw_mode()?;
    Ok(())
}
