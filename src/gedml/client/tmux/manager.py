import libtmux

def clear_tmux(session_name):
    """Kill the specified tmux session.

    Args:
        session_name(str):
            The name of tmux session to be killed.
    """
    target_session = TmuxManager(session_name=session_name)
    target_session.close_all()
    del target_session

class TmuxManager:
    """Class to manage the Tmux software.

    Note:
        Tmux must be installed before.

    Args:
        session_name(str):
            The name of tmux session.
        conda_env(str):
            The name of conda environment for window intialization.

    Example:
        >>> manager = TmuxManager()
        >>> manager.new_window(window_name="test")
        >>> cmd = "clear"
        >>> manager.send_cmd(cmd)
    
    Attributes:
        session:
            Current tmux session pointer.
        curr_window:
            Current tmux window pointer.
        curr_pane:
            Current tmux pane pointer by which user can send command.
    """
    def __init__(
        self,
        session_name='borel',
        conda_env="pytorch",
    ):
        self.session_name = session_name
        self.conda_env = conda_env

        self.initiate_manager()
    
    def initiate_manager(self):
        """Initiate libtmux.Server and create specified session.
        """
        self.server = libtmux.Server()
        # initiate session
        self.session = self.server.find_where(
            {
                "session_name": self.session_name
            }
        )
        if self.session is None:
            self.session = self.server.new_session(
                session_name=self.session_name
            )
        
        self.window_list = self.session.list_windows()
        self.curr_window = self.window_list[0]
        self.curr_pane = self.curr_window.list_panes()[0]
    
    def new_window(self, window_name=None, *args):
        """Create a new tmux window.

        After creating the tmux window, conda initialization command will be sent to enter into the specified conda environment.

        Args:
            window_name(str):
                The name of window to be created.
        """
        self.curr_window = self.session.new_window(
            window_name=window_name,
            attach=False,
        )
        self.window_list = self.session.list_windows()
        self.curr_pane = self.curr_window.list_panes()[0]
        # to conda environment
        cmd = "conda activate {}".format(self.conda_env)
        self.curr_pane.send_keys(cmd)
    
    def send_cmd(self, cmd):
        """Send command to the current tmux pane.

        Args:
            cmd(str):
                Command to be sent.
        """
        self.curr_pane.send_keys(cmd)
    
    def close_all(self):
        """Close all tmux window.
        """
        for w in self.window_list:
            w.kill_window()


    

    
