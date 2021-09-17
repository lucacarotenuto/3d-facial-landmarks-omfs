from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
import torch


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    total_error = 0
    error = 0
    for i, data in enumerate(dataset):
        model.set_input(data)
        error, nexamples = model.test()
        writer.update_counter(error, nexamples)
        total_error = torch.sum(error)
        print("total error" + str(total_error))
    writer.print_acc(epoch, error)
    return total_error
    #return writer.acc


if __name__ == '__main__':
    run_test()
