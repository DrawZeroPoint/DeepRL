from tensorboardX import SummaryWriter


class TensorboardInterface:

    def __init__(self, path, name):
        self.writer = SummaryWriter()
        self.path = path
        self.name = name

    def add_scalar(self, name, val, count):
        self.writer.add_scalar(name, val, count)

    def export(self):
        self.writer.export_scalars_to_json(self.path + self.name + ".json")
        self.writer.close()

