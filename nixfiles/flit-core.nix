{
  buildPythonPackage,
  fetchPypi,
}:
buildPythonPackage rec {
  pname = "flit_core";
  version = "3.7.1";
  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-FJVa80DEMDXb+pa17kdAfjd+4zf2nnD3MGSUDSfQpE8=";
  };
}
