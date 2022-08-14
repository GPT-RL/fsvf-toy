{
  buildPythonPackage,
  fetchPypi,
  numpy,
  setuptools-scm,
}:
buildPythonPackage rec {
  pname = "tensorstore";
  version = "0.1.22";
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-qQqytAtjXtThWh+a+dNLeQHlxYUGxdUOgK69aFDQtwA=";
  };
  propagatedBuildInputs = [numpy];
}
