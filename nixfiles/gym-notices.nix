{
  buildPythonPackage,
  fetchPypi,
}:
buildPythonPackage rec {
  pname = "gym-notices";
  version = "0.0.7";
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-JUtmdBLb4gUTUhNZXJY82DBmg4LOV7aKDCgqI47rNbk=";
  };
}
