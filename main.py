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


if __name__ == '__main__':
    app.run(main)