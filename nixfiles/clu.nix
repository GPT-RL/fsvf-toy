{
  buildPythonPackage,
  fetchPypi,
  pkgs,
}:
buildPythonPackage rec {
  pname = "clu";
  version = "0.0.7";
  src = builtins.fetchTarball {
    sha256 = "sha256:0scpcw5vylg2f3fd8hgwr6zkcwl8kwq1m16p3yh6dqrcay3xx38p";
    url = "https://files.pythonhosted.org/packages/fc/dc/8dbb7f562c9d684b78be44b091d277775461dcdc9f3fce3732674c89b36b/clu-0.0.7.tar.gz";
  };
}
