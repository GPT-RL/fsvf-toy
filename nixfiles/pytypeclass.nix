{
  lib,
  buildPythonPackage,
  fetchPypi,
}:
buildPythonPackage rec {
  pname = "pytypeclass";
  version = "0.1.1";
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-BzX9vS7fjoBLiorI2+xOqPz5PPhm5S0Kxcpg2QrI28g=";
  };

  meta = with lib; {
    maintainers = with maintainers; [ethanabrooks];
  };
}
