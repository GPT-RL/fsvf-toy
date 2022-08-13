{
  lib,
  buildPythonPackage,
  fetchPypi,
}:
buildPythonPackage rec {
  pname = "gym-minigrid";
  version = "1.1.0";
  src = fetchTarball {
    url = "https://files.pythonhosted.org/packages/12/01/a9b735ca3b6ab12a9f899ea9c64cd88e06c095d1e1cef2af6879617455df/gym_minigrid-1.1.0.tar.gz";
    sha256 = "sha256:0fpxfha2y76gw80hndbixp8i1a4mdb28gban04l0mqqx0rzd0vrk";
  };
}
