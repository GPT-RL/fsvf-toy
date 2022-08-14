{
  buildPythonPackage,
  fetchPypi,
  packaging,
  pytest,
  tomli,
  typing-extensions,
}:
buildPythonPackage rec {
  pname = "setuptools_scm";
  version = "7.0.5";
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-Ax4Tr3cdb4krlBrbbqBFRbv5Hrxc5ox4qvP/9uH7SEQ=";
  };
  buildInputs = [
    packaging
    tomli
    typing-extensions
    pytest
  ];
}
