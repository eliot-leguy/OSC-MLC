__all__ = ["transition_time"]


class transition_time:
    def __init__(self):
        self.nb_back_to_normal = 0
        self.nb_go_to_aware = 0
        self.current_task = 0
        self.nb_instance = 0

    # Update function. Takes target and predicted label vectors and increments the accuracy sum
    def update(self, target: dict, predicted: dict, task: int):
        if task == 0:
            if self.current_task != 0:
                self.current_task = 0
                self.nb_back_to_normal = 0
                self.nb_instance = 0
            self.nb_instance += 1
            if (
                target["L4"] != predicted["L4"]
                or target["L5"] != predicted["L5"]
                or target["L6"] != predicted["L6"]
            ):
                self.nb_back_to_normal = self.nb_instance

        elif task == 1:
            if self.current_task != 1:
                self.current_task = 1
                self.nb_go_to_aware = 0
                self.nb_instance = 0
            self.nb_instance += 1
            if (
                target["L4"] != predicted["L4"]
                or target["L5"] != predicted["L5"]
                or target["L6"] != predicted["L6"]
            ):
                self.nb_go_to_aware = self.nb_instance

        elif task == 2:
            if self.current_task != 2:
                self.current_task = 2
                self.nb_go_to_aware = 0
                self.nb_instance = 0
            self.nb_instance += 1
            if (
                target["L4"] != predicted["L4"]
                or target["L5"] != predicted["L5"]
                or target["L6"] != predicted["L6"]
            ):
                self.nb_go_to_aware = self.nb_instance

    # Calculates and return the current average accuracy of evaluation for the metric
    def get(self):
        if self.current_task == 0:
            return {0: self.nb_back_to_normal}

        else:
            return {self.current_task: self.nb_go_to_aware}
