from abc import abstractmethod

from bmipy import BmiExtension


class Xmi(BmiExtension):
    """
    The eXtended Model Interface, the BMI extension ``"xmi"``.

    Request ``"xmi"`` at ``initialize`` and get this object from
    ``get_extension("xmi")``. It exposes the model's outer convergence loop,
    and sub-components that share the time stepping but have their own
    convergence loop.
    """

    @abstractmethod
    def prepare_time_step(self, dt: float) -> None:
        """Prepare a single time step.

        Read data from input files and calculate the current time step length
        and the simulation time at the end of the current time step.

        Parameters
        ----------
        dt : float
            Model time step length for the current time step.
        """
        ...

    @abstractmethod
    def do_time_step(self) -> None:
        """Perform a single time step.

        Build and solve a time step to completion. This method encapsulates
        the prepare_solve, solve, and finalize_solve methods.
        """
        ...

    @abstractmethod
    def finalize_time_step(self) -> None:
        """Finalize the time step.

        Write messages and output after model convergence has
        been achieved.
        """
        ...

    @abstractmethod
    def get_subcomponent_count(self) -> int:
        """Get the number of components in the simulation.

        For most applications, this number will be equal to 1.

        Returns
        -------
        int
          The number of components in the simulation.
        """
        ...

    @abstractmethod
    def prepare_solve(self, component_id: int) -> None:
        """Prepare for solving the system of equations.

        This preparation mostly consists of advancing model states.

        Parameters
        ----------
        component_id : int
            Component id number.
        """
        ...

    @abstractmethod
    def solve(self, component_id: int) -> bool:
        """Build and solve the linear system of equations.

        Formulate the system of equations for this outer (Picard) iteration and
        solve. New data used when formulating the system of equations can be
        injected prior to solve().  Before calling this, a matching call to
        prepare_solve() should be performed.

        Parameters
        ----------
        component_id : int
            Component id number.

        Returns
        -------
        bool
          Boolean indicating if convergence has been achieved.

        """
        ...

    @abstractmethod
    def finalize_solve(self, component_id: int) -> None:
        """Finalize the model solve for the current time step.

        Finalize model variables prior to calling finalize_time_step(). This
        method should always be called after calls to prepare_solve() and
        solve()

        Parameters
        ----------
        component_id : int
            Component id number.

        """
        ...

    @abstractmethod
    def get_version(self) -> str:
        """Get the version of the kernel."""
        ...

    @abstractmethod
    def get_var_address(
        self, var_name: str, component_name: str, subcomponent_name: str = ""
    ) -> str:
        """Get the address of a given variable

        Parameters
        ----------
        var_name : str
            The variable name
        component_name : str
            The name of the component
        subcomponent_name : str, optional
            If applicable the name of the subcomponent, by default ""

        Returns
        -------
        str
            The variable address
        """
        ...
