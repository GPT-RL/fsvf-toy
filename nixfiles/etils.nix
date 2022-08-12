{
  pkgs,
  pyprev,
  pyfinal,
}:
pyprev.buildPythonPackage rec {
  pname = "etils";
  version = "0.6.0";
  format = "flit";
  src = pkgs.fetchFromGitHub {
    owner = "google";
    repo = pname;
    rev = "v${version}";
    hash = "sha256-O3ApPfsAbha6OS7ruUVdkXXkfo/M2CNIDjX6QXlOzXY=";
  };
  passthru.optional-dependencies = with pyfinal; rec {
    array-types = enp;
    ecolab =
      [
        jupyter
        numpy
        mediapy
      ]
      ++ enp
      ++ epy;
    edc = epy;
    enp = [numpy] ++ epy;
    epath =
      [
        importlib-resources
        zipp
      ]
      ++ epy;
    epy = [typing-extensions];
    etqdm = [absl-py tqdm] ++ epy;
    etree = builtins.concatLists [array-types epy enp etqdm];
    etree-dm = [dm-tree] ++ etree;
    etree-jax = [jaxWithoutCuda] ++ etree;
    etree-tf = [tensorflow] ++ etree;
    all = builtins.concatLists [
      array-types
      ecolab
      edc
      enp
      epath
      epy
      etqdm
      etree
      etree-dm
      etree-jax
      etree-tf
    ];
  };
}
