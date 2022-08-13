{
  buildPythonPackage,
  fetchPypi,
}:
buildPythonPackage rec {
  pname = "gym-notices";
  version = "0.0.7";
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-BzX9vS7fjoBLiorI2+xOqPz5PPhm5S0Kxcpg2QrI28g=";
  };
}
