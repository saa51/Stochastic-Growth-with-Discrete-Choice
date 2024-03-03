import gin
from vfi_solver import SOGVFISolver
from absl import flags, app


flags.DEFINE_multi_string(
  'gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string(
'gin_param', None, 'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS

def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    solver = SOGVFISolver()
    print(solver.steady_state())
    res = solver.solve()
    res.plot_value(show=True)
    res.plot_q(idxa_vec=[0, 2, 4], show=True)
    res.plot_policy(idxa_vec=[0, 2, 4], show=True)
    res.plot_capital_diff(idxa_vec=[0, 2, 4], show=True)
    res.plot_value_derivative(show=True)
    res.plot_value_2derivative(show=True)
    res.plot_policy_derivative(idxa_vec=[0, 2, 4], show=True)
    res.plot_policy_2derivative(idxa_vec=[0, 2, 4], show=True)

if __name__ == '__main__':
    app.run(main)