{
  buildPythonPackage,
  fetchPypi,
}:
buildPythonPackage rec {
  pname = "setuptools-scm";
  version = "7.0.5";
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-BzX9vS7fjoBLiorI2+xOqPz5PPhm5S0Kxcpg2QrI28g=";
  };
}
