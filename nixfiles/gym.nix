{
  # python deps
  buildPythonPackage,
  cloudpickle,
  gym-notices,
  numpy,
  pytest,
  # pkgs deps
  fetchFromGitHub,
}:
buildPythonPackage rec {
  pname = "gym";
  version = "0.21.0";
  src = fetchFromGitHub {
    owner = "ethanabrooks";
    repo = "gym";
    rev = "559d8127b47d53f479ad738789383660533f2a3c";
    sha256 = "sha256-2OLF5CgXWxCLeeyGSrPgg5Xo+ztNOM3W/lm0ea2/LuE=";
  };
  propagatedBuildInputs = [
    cloudpickle
    gym-notices
    numpy
    pytest
  ];
}
