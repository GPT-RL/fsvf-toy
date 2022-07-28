{
  pkgs,
  pyprev,
  pyfinal,
}:
pyprev.tensorflow-datasets.overridePythonAttrs (old: rec {
  doCheck = false;
  version = "4.6.0";
  src = pyprev.fetchPypi {
    pname = "tensorflow-datasets";
    inherit version;
    sha256 = "sha256-naKmUpfpgfYcXxy0VpYGa5xAE9l4A6bfdDgrwk4hN8s=";
  };
  pythonImportsCheck = null;
  propagatedBuildInputs =
    old.propagatedBuildInputs
    ++ pyfinal.etils.optional-dependencies.epath
    ++ [pyfinal.etils];
})
