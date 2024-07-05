import gin
from vfi_solver import SOGVFISolver
from absl import flags, app
import pickle
import datetime
from pathlib import Path
import shutil


flags.DEFINE_multi_string(
  'gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string(
'gin_param', None, 'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS

def main(argv):
    idxa_vec = [4, 9, 14]
    st_time = datetime.datetime.now()
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    solver = SOGVFISolver()
    print(solver.steady_state())
    res = solver.solve()
    res.plot_value(show=True)
    res.plot_q(idxa_vec=idxa_vec, show=True)
    res.plot_policy(idxa_vec=idxa_vec, show=True)
    res.plot_capital_diff(idxa_vec=idxa_vec, show=True)
    res.plot_value_derivative(show=True)
    res.plot_value_2derivative(show=True)
    res.plot_policy_derivative(idxa_vec=idxa_vec, show=True)
    res.plot_policy_2derivative(idxa_vec=idxa_vec, show=True)
    res.plot_l(idxa_vec=idxa_vec, show=True)
    res.plot_exact_value(a=0, simu_num=1000, periods=1000, show=True)

    res_path = Path('./results') / st_time.strftime("%Y%m%d-%H%M")
    res_path.mkdir(parents=True, exist_ok=True)
    with open(res_path / "vfi.pkl", 'wb') as f:
        pickle.dump(res.to_dict(), f)
    for f in FLAGS.gin_file:
      shutil.copy(f, res_path / Path(f).name)

if __name__ == '__main__':
    app.run(main)