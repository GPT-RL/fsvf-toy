{
  pkgs,
  pyprev,
  pyfinal,
}:
pyprev.dm-tree.overridePythonAttrs rec {
  meta.broken = false;
  pname = "dm-tree";
  version = " 0.1.7 ";
  format = "wheel";
  src = pyprev.fetchPypi {
    inherit pname version format;
    dist = pyfinal.python;
  };
}
