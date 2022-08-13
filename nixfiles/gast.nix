{
  buildPythonPackage,
  fetchPypi,
}:
buildPythonPackage rec {
  pname = "gast";
  version = "0.5.3";
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-z76iWCDmU6+cfRgH9lnOCgqcZPJDlCGnu6TwmD9TLeo=";
  };
}
