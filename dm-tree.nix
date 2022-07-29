{
  pkgs,
  pyprev,
  pyfinal,
}:
pyprev.dm-tree.overridePythonAttrs rec {
  doCheck = false;
  meta.broken = false;
  pname = "dm-tree";
  pythonImportsCheck = [];
  version = "0.1.7";
  dontUseSetuptoolsCheck = true;
  src = pyprev.fetchPypi {
    inherit pname version;
    sha256 = "sha256-MP7IrKW5KCPA55ai8zuHW03M1HC1fpHmxUJAXF93/So=";
  };
}
