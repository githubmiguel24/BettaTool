import { RulerIcon, AngleIcon, ProtractorIcon } from "../components/Icons.jsx";

export const measurements = [
  {
    key: "caudal-spread-angle",
    label: "Caudal Spread Angle",
    description: "Angle formed by the flared caudal fin edges",
    icon: ProtractorIcon,
  },
  {
    key: "dorsal-body-ratio",
    label: "Dorsal Fin / Body Ratio Measurement",
    description: "Dorsal fin height relative to body length",
    icon: RulerIcon,
  },
  {
    key: "anal-body-ratio",
    label: "Anal Fin / Body Ratio Measurement",
    description: "Anal fin height relative to body length",
    icon: RulerIcon,
  },
  {
    key: "caudal-body-ratio",
    label: "Caudal Fin / Body Ratio Measurement",
    description: "Caudal fin span relative to body length",
    icon: AngleIcon,
  },
  {
    key: "anal-caudal-ratio",
    label: "Anal Fin / Caudal Fin Ratio",
    description: "Anal fin height relative to caudal fin span",
    icon: RulerIcon,
  },
  {
    key: "dorsal-caudal-ratio",
    label: "Dorsal Fin / Caudal Fin Ratio",
    description: "Dorsal fin height relative to caudal fin span",
    icon: AngleIcon,
  },
];
