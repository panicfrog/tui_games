pub trait Env {
    type State;
    type Action;
    type Status;

    // reset the environment to its initial state
    fn reset(&mut self) -> Self::State;
    // take an action and return the next state and status
    fn step(&mut self, action: Self::Action) -> (Self::State, Self::Status);
    // get the current state of the environment
    fn current_state(&self) -> Self::State;
    // get the legal actions for the current state or a given state and action to return spacific actions
    // if state_and_action is None, return the legal actions for the current state
    fn legal_actions(
        &self,
        state_and_action: Option<(Self::State, Self::Action)>,
    ) -> Vec<Self::Action>;
    // check if the environment is in a terminal state
    fn is_terminal(&self) -> bool;
    // check if the environment is in a win state
    fn is_win(&self) -> bool;
}

pub use rand;
