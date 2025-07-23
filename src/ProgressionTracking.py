import torch

class ProgressionTracking():

    def __init__(self, metric_names, tracking_frequency):
        """
        ...

        Parameters
        ----------
        metric_names : list of strs
            List of strings defining the metrics of interest.
        tracking_frequency : int
            Frequency progression tracking for solving (solve_Phi_...())
        """

        implemented_metrics = ['dT',
                               'summed_abs_change',
                               'summed_abs_error',
                               'cross_section_along_x',
                               ]

        assert set(metric_names).issubset(set(implemented_metrics)), 'Not all input metrics have been implemented.'

        self.metric_functions = {metric_name : self._get_function_for_metric(metric_name) for metric_name in metric_names}
        self.metrics = {metric_name : [] for metric_name in metric_names}
        self.metrics['tracking_frequency'] = tracking_frequency
        self.tracking_frequency = tracking_frequency #TODO: Make dynamic (eg. depending on chnage in summed_abs_change)

    def update_tracked_metrics(self):
        """
        Updates the dict self.metrics by calling all functions in the
        self.metric_functions dict.
        """

        for metric_name, metric_function in self.metric_functions.items():
            intermediate = metric_function()
            self.metrics[metric_name].append(intermediate)
            del intermediate

    def _get_function_for_metric(self, metric_name):
        """
        Takes a str as input and returns the corresponding callable function.

        Parameters
        ----------
        metric_name : str
            Name of metric

        Returns
        -------
        function : function
            The function corresponding to metric_name.
        """

        if metric_name == 'summed_abs_change':
            return self.get_summed_abs_change
        elif metric_name == 'dT':
            return self.get_dT
        elif metric_name == 'summed_abs_error':
            return self.get_summed_abs_error
        elif metric_name == 'cross_section_along_x':
            return self.get_cross_section_along_x
        else:
            return None

    def get_summed_abs_change(self):
        """
        """

        sac = float(torch.sum(torch.abs(self.Phi - self.Phi_next), dtype=torch.float32))

        return sac

    def get_summed_abs_error(self):
        """
        Only valid when analytical solution is known.
        """

        idx_z = self.z_res // 2

        mse = torch.sum(torch.abs(self.analytical_solution - self.compute_B_from_Phi()[:, :, idx_z]))

        return mse.item()

    def get_cross_section_along_x(self):
        """
        Don't use this funtion. It takes up a lot of memory.
        """

        idx_z = self.z_res // 2
        idx_y = self.xy_res // 2

        cross_section = self.compute_B_from_Phi()[idx_y, :, idx_z]

        return cross_section

    def get_cross_section_along_y(self):
        """
        Don't use this funtion. It takes up a lot of memory.
        """

        idx_z = self.z_res // 2
        idx_x = self.xy_res // 2

        cross_section = self.compute_B_from_Phi()[:, idx_x, idx_z].numpy()

        return cross_section

    def get_dT(self):
        """
        """

        return self.dT
