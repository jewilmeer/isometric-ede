// Isometric scene of Helenaveen, rendered tile by tile by render/run.mjs.
// World units are meters: x east, y up, z south, origin at the center of the 3 x 3 km area.
import * as THREE from 'three';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';

THREE.ColorManagement.enabled = false; // colors are painted in sRGB and used as-is, like pixel art

const SUN_AZIMUTH = 250; // keep in sync with scripts/sun.py
const SUN_ELEVATION = 40;
const SLAB = 14; // depth of the diorama block under the ground
const CAMERA_ELEVATION = 30; // 2:1 pixel-art isometric
const TOP_MARGIN = 32;
const BACKGROUND = new THREE.Color(0x20242c);
const ROAD_ALPHA = 0.5; // marks road pixels in the color target, see makeRoads and the post pass

// The era's data (render/run.mjs --era): data/build for now, data/build-<era> for an earlier year.
const DATA = new URLSearchParams(location.search).get('data') || '/data/build';
const TYPES = { house: 0, barn: 1, shed: 2, greenhouse: 3, tank: 4, silo: 5, canopy: 6, church: 7, tower: 8, turret: 9, school: 10, mast: 11, cross: 12, bridge: 13, chimney: 14, station: 15, railing: 16, canteen: 17, hall: 18, station_hall: 19, station_tower: 20, factory: 21, stoer: 22 , dome: 23, mausoleum: 24, station1902: 25, ruin: 26, parking: 27, cinema: 28, mosque: 29, loods: 30 };

const rgb = (r, g, b) => new THREE.Color(r / 255, g / 255, b / 255);
const hash = (x) => {
  const s = Math.sin(x * 12.9898) * 43758.5453;
  return s - Math.floor(s);
};

const WALLS_BARN = [
  [rgb(108, 122, 104), 3], [rgb(78, 96, 80), 2], [rgb(150, 146, 132), 2], [rgb(120, 70, 54), 2], [rgb(96, 102, 108), 2],
];
// Roof colors snap to this palette after a little contrast from the aerial sample.
const ROOFS = [
  rgb(176, 84, 58), rgb(140, 64, 50), rgb(108, 56, 46), rgb(66, 68, 74), rgb(88, 90, 96), rgb(120, 122, 124),
  rgb(150, 150, 146), rgb(184, 182, 176), rgb(120, 92, 70), rgb(98, 114, 98), rgb(214, 214, 206), rgb(44, 48, 56),
];

function pickWeighted(list, r) {
  const total = list.reduce((s, [, w]) => s + w, 0);
  let t = r * total;
  for (const [c, w] of list) if ((t -= w) <= 0) return c;
  return list[0][0];
}

function nearestRoof(c) {
  let best = ROOFS[0], bd = Infinity;
  for (const p of ROOFS) {
    const d = (p.r - c.r) ** 2 * 0.9 + (p.g - c.g) ** 2 * 1.2 + (p.b - c.b) ** 2;
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}

// Facades by construction year (BAG): the eras of a Peel village.
const ERAS = [
  [1945, [[rgb(138, 68, 50), 4], [rgb(118, 58, 44), 3], [rgb(158, 90, 64), 2], [rgb(226, 220, 206), 1]], rgb(236, 234, 226)],
  [1970, [[rgb(160, 88, 62), 4], [rgb(176, 106, 74), 3], [rgb(140, 78, 58), 2], [rgb(200, 172, 132), 1]], rgb(232, 230, 222)],
  [1995, [[rgb(112, 72, 56), 3], [rgb(128, 84, 62), 3], [rgb(196, 170, 128), 2], [rgb(150, 80, 58), 2]], rgb(88, 62, 48)],
  [9999, [[rgb(150, 80, 60), 3], [rgb(112, 108, 104), 2], [rgb(226, 222, 212), 2], [rgb(84, 64, 52), 2], [rgb(176, 104, 74), 1]], rgb(58, 60, 64)],
];

function era(m) {
  return ERAS.find(([until]) => (m.year || 1985) < until);
}

function frameColor(m) {
  if (m.frame_fixed) return rgb(...m.frame_fixed);
  if (m.kind === 'school') return rgb(176, 40, 34);
  return era(m)[2];
}

function roofColor(m) {
  if (m.roof_fixed) return rgb(...m.roof_fixed);
  if (!m.roof_rgb) return ROOFS[4];
  const c = rgb(...m.roof_rgb);
  // Push saturation up a bit: the photo is hazy, pixel art is not.
  const hsl = {};
  c.getHSL(hsl);
  c.setHSL(hsl.h, Math.min(1, hsl.s * 1.5), hsl.l);
  return nearestRoof(c);
}

function wallColor(m) {
  if (m.wall_fixed) return rgb(...m.wall_fixed);
  if (m.kind === 'tank') return [rgb(74, 98, 82), rgb(150, 150, 140), rgb(64, 72, 80)][Math.floor(m.rand * 3)];
  if (m.kind === 'silo') return rgb(196, 198, 192);
  if (m.kind === 'barn' || m.kind === 'shed') return pickWeighted(WALLS_BARN, m.rand);
  return pickWeighted(era(m)[1], m.rand);
}

async function fetchF32(url) {
  return new Float32Array(await (await fetch(url)).arrayBuffer());
}

// ---------------------------------------------------------------------------------------------
// Buildings

const buildingVertex = `
attribute vec2 aUv;
attribute float aKind;
attribute float aType;
attribute float aAxis;
attribute vec3 aFrame;
attribute vec2 aAtlas;
varying vec3 vFrame;
varying vec2 vAtlas;
varying vec2 vUv2;
varying float vKind;
varying float vType;
varying float vAxis;
varying vec3 vWorld;
varying vec3 vWorldNormal;
`;

const buildingFragment = `
varying vec2 vUv2;
varying float vKind;
varying float vType;
varying float vAxis;
varying vec3 vWorld;
varying vec3 vWorldNormal;
varying vec3 vFrame;
varying vec2 vAtlas;
uniform sampler2D roofAtlas;

vec3 buildingPattern(vec3 base) {
  vec3 c = base;
  bool wall = vKind > 0.5 && vKind < 1.5;
  bool roof = vKind > 1.5;
  int type = int(vType + 0.5);
  if (wall) {
    if (type == 0) {
      // Houses: a grid of windows per storey, and a darker plinth.
      float fl = mod(vUv2.y, 2.8);
      float col = mod(vUv2.x + 0.4, 3.1);
      bool win = vUv2.y > 0.5 && fl > 0.85 && fl < 2.15 && col > 0.8 && col < 2.2;
      if (win) {
        c = vec3(0.21, 0.26, 0.33);
        if (fl > 1.95 || col < 0.98) c = vFrame;
      }
      if (vUv2.y < 0.35) c *= 0.72;
    } else if (type == 1 || type == 2) {
      // Barns and sheds: corrugated metal or brick with a concrete plinth.
      if (mod(vUv2.x, 1.2) < 0.3) c *= 0.88;
      if (vUv2.y < 0.9) c = mix(c, vec3(0.6, 0.6, 0.58), 0.6);
      float bay = mod(vUv2.x, 14.0);
      if (type == 1 && bay > 4.0 && bay < 8.0 && vUv2.y < 4.2 && vUv2.y > 0.2) c = vec3(0.36, 0.38, 0.4);
    } else if (type == 3) {
      // Greenhouse glass: pale with the crop showing through, white mullions.
      c = mix(vec3(0.74, 0.82, 0.84), vec3(0.42, 0.58, 0.42), 0.3);
      if (mod(vUv2.x, 2.0) < 0.28 || mod(vUv2.y, 1.7) < 0.22) c = vec3(0.9, 0.92, 0.92);
    } else if (type == 4) {
      if (mod(vUv2.y, 1.2) < 0.18) c *= 0.85;
    } else if (type == 5) {
      if (mod(vUv2.y, 1.6) < 0.2) c *= 0.82;
    } else if (type == 6) {
      // Canopy: mostly open, just posts.
      if (vUv2.y < 2.6 && mod(vUv2.x, 4.0) > 0.4) c = vec3(0.28, 0.27, 0.25);
    } else if (type == 7) {
      // Church: tall round-arched windows in bays, a darker plinth.
      float cx = mod(vUv2.x + 1.0, 4.2) - 2.1;
      float y = vUv2.y;
      bool arch = y > 6.0 && length(vec2(cx, y - 6.0)) < 0.75;
      if (abs(cx) < 0.75 && y > 2.2 && (y < 6.0 || arch)) {
        c = vec3(0.22, 0.25, 0.32);
        if (mod(y, 1.1) < 0.12 || abs(cx) < 0.06) c = vec3(0.55, 0.56, 0.58);
      }
      if (y < 0.6) c *= 0.78;
    } else if (type == 8) {
      // Church tower: louvred belfry, a clock face below it, a slit window, brick plinth.
      float u = vUv2.x, y = vUv2.y, top = vAxis;
      if (abs(u) < 0.8 && y > top - 3.8 && y < top - 0.9) {
        c = vec3(0.16, 0.16, 0.18);
        if (mod(y, 0.42) < 0.14) c = vec3(0.42, 0.36, 0.3);
      }
      float r = length(vec2(u, y - (top - 5.4)));
      if (r < 0.85) c = r < 0.1 ? vec3(0.15) : (r > 0.72 ? vec3(0.2, 0.2, 0.22) : vec3(0.93, 0.92, 0.86));
      if (abs(u) < 0.28 && y > 4.0 && y < 8.5) c = vec3(0.2, 0.22, 0.28);
      if (y < 0.6) c *= 0.78;
    } else if (type == 9) {
      // Roof turret: white, with arched openings near the top.
      if (abs(vUv2.x) < 0.22 && vUv2.y > vAxis - 1.9 && vUv2.y < vAxis - 0.4) c = vec3(0.18, 0.18, 0.2);
    } else if (type == 10) {
      // 1930s school: tall classroom windows with red frames.
      float col = mod(vUv2.x, 2.0);
      float y = vUv2.y;
      if (col > 0.45 && col < 1.55 && y > 0.9 && y < 3.7) {
        c = vec3(0.22, 0.25, 0.3);
        if (col < 0.58 || col > 1.42 || y < 1.02 || y > 3.58 || abs(y - 2.9) < 0.08) c = vFrame;
      }
      if (y < 0.5) c *= 0.8;
    } else if (type == 13) {
      // Bridge: smooth concrete with a joint every few meters; railings plain.
      if (mod(vUv2.x, 6.0) < 0.12) c *= 0.9;
    } else if (type == 15) {
      // Edge of the station roof: a thin grey fascia.
    } else if (type == 19) {
      // Station hall: a glass wall in tall panes between slim grey mullions.
      c = vec3(0.24, 0.29, 0.34);
      if (mod(vUv2.x, 1.8) < 0.1 || mod(vUv2.y, 3.5) < 0.12) c = vec3(0.62, 0.64, 0.66);
      if (vUv2.y > vAxis - 0.6) c = vec3(0.55, 0.57, 0.6);
    } else if (type == 20) {
      // The station tower: light grey cladding in fine vertical seams.
      if (mod(vUv2.x, 0.55) < 0.07) c *= 0.88;
    } else if (type == 21) {
      // Enka factory (1920s): dark brick bays between pilasters, red steel windows, a light concrete cornice. The tall
      // main building has storeys of paired windows; the halls tall windows.
      float bay = mod(vUv2.x, 4.2), y = vUv2.y, top = vAxis;
      if (bay < 0.6) c *= 1.12;
      if (top > 14.0) {
        float fl = mod(y, 3.4);
        float w = bay - 0.6;
        bool win = y > 1.0 && y < top - 1.4 && fl > 1.0 && fl < 2.9 && ((w > 0.5 && w < 1.6) || (w > 2.0 && w < 3.1));
        if (win) {
          c = vec3(0.24, 0.28, 0.33);
          if (fl < 1.1 || fl > 2.8 || mod(w, 1.5) < 0.08 || abs(fl - 1.9) < 0.05) c = vFrame;
        }
      } else if (bay > 1.0 && bay < 3.8 && y > 1.2 && y < top - 1.6) {
        c = vec3(0.24, 0.28, 0.33);
        if (mod(bay - 1.0, 0.93) < 0.08 || mod(y - 1.2, 1.1) < 0.08) c = vFrame;
      }
      if (y > top - 0.6) c = vec3(0.8, 0.78, 0.72);
      if (y < 0.5) c *= 0.8;
    } else if (type == 22) {
      // Stoer: near-black brick, each window with a yellow panel beside it; grey shutters on the ground floor.
      float bay = mod(vUv2.x, 3.2), fl = mod(vUv2.y, 3.0);
      if (fl > 0.6 && fl < 2.5) {
        if (vUv2.y < 3.0 && bay > 0.3 && bay < 2.9) c = vec3(0.42, 0.44, 0.47);
        else if (bay > 0.3 && bay < 1.4) c = vec3(0.26, 0.3, 0.34);
        else if (bay > 1.6 && bay < 2.9) c = vFrame;
      }
    } else if (type == 16) {
      // Railing: a top rail and a bottom rail, with balusters between; open in between so the children show through.
      float y = vUv2.y - (vAxis - 1.1);
      bool bar = y > 1.0 || y < 0.12 || mod(vUv2.x, 0.45) < 0.07;
      if (!bar) discard;
    } else if (type == 17) {
      // The Enka canteen (1950s): a concrete grid of bays, each a tall window over a white panel, green frames.
      float col = mod(vUv2.x, 3.3), fl = mod(vUv2.y, 3.4);
      if (col > 0.45 && fl > 0.45 && vUv2.y > 0.4) {
        c = fl < 1.3 ? vec3(0.94, 0.94, 0.91) : vec3(0.56, 0.64, 0.7);
        if (abs(col - 1.9) < 0.07 || abs(fl - 2.4) < 0.07 || abs(fl - 1.3) < 0.06) c = vFrame;
      }
    } else if (type == 18) {
      // Factory hall: brick, a band of tall steel windows, a darker plinth.
      float col = mod(vUv2.x, 4.0);
      if (col > 0.9 && col < 3.1 && vUv2.y > 1.4 && vUv2.y < 4.2) {
        c = vec3(0.3, 0.33, 0.36);
        if (mod(vUv2.y - 1.4, 0.7) < 0.07 || mod(col - 0.9, 0.55) < 0.06) c = vec3(0.5, 0.52, 0.5);
      }
      if (vUv2.y < 0.6) c *= 0.8;
    } else if (type == 23) {
      // The Stingerbol: white panels with faint seams.
      if (mod(vUv2.x, 3.0) < 0.08 || mod(vUv2.y, 2.5) < 0.08) c *= 0.92;
    } else if (type == 24) {
      // The Mausoleum: brick, two rows of twenty light stone plates with gold names, an inscription band above.
      float u = mod(vUv2.x + 0.3, 1.3), y = vUv2.y;
      if (u > 0.15 && u < 1.15 && ((y > 0.9 && y < 1.8) || (y > 2.0 && y < 2.9))) {
        c = vec3(0.86, 0.83, 0.76);
        if (mod(y, 0.18) < 0.05 && u > 0.3 && u < 1.0) c = vec3(0.78, 0.66, 0.36);
      }
      if (y > 3.25 && y < 3.6) c = vec3(0.84, 0.8, 0.72);
      if (y < 0.4) c *= 0.8;
    } else if (type == 25) {
      // Station Ede Centrum (1902): brick with tall round-headed windows, light stone bands and a darker plinth.
      float cx = mod(vUv2.x + 1.4, 2.8) - 1.4, y = vUv2.y;
      float sy = y - 0.9;
      bool arch = sy > 1.85 && length(vec2(cx, sy - 1.85)) < 0.55;
      if (abs(cx) < 0.55 && sy > 0.0 && (sy < 1.85 || arch)) {
        c = vec3(0.2, 0.23, 0.28);
        if (abs(cx) < 0.04 || abs(sy - 1.5) < 0.04) c = vFrame;
        if (length(vec2(cx, sy - 1.85)) > 0.47 && (arch || sy < 0.05)) c = vFrame;
        if (abs(cx) > 0.47) c = vFrame;
      }
      if (abs(cx) < 0.7 && sy > 2.4 && sy < 2.55 + 0.0 && !arch) c = vec3(0.8, 0.77, 0.7);
      if (y > 0.5 && y < 0.7) c = vec3(0.8, 0.77, 0.7);
      if (y > vAxis - 0.5 && y < vAxis - 0.15) c = vec3(0.8, 0.77, 0.7);
      if (y < 0.5) c *= 0.7;
    } else if (type == 26) {
      // A free-standing factory wall: two storeys of empty window openings between brick piers, the sky showing
      // through.
      float col = mod(vUv2.x, 4.2), y = vUv2.y;
      bool hole = col > 0.7 && col < 3.5 && ((y > 1.2 && y < 4.0) || (y > 5.2 && y < 8.0));
      if (hole) discard;
      if (y > vAxis - 0.5) c = vec3(0.82, 0.8, 0.74);
    } else if (type == 27) {
      // Parking garage: open decks every 3 m between concrete edge beams, cars in the dark gaps.
      float fl = mod(vUv2.y, 3.0);
      if (fl > 0.9 && fl < 2.5 && vUv2.y < vAxis - 0.5) {
        c = vec3(0.22, 0.23, 0.25);
        if (mod(vUv2.x, 2.6) < 1.6 && fl < 1.6) c = vec3(0.45, 0.47, 0.52);
      }
    } else if (type == 28) {
      // Cinema (Pathe Ede): large closed facades, a glass entrance zone at the foot, a few horizontal grooves and a
      // light band at the top.
      float y = vUv2.y, top = vAxis;
      if (y < 3.8) {
        float col = mod(vUv2.x, 2.4);
        if (y > 0.3 && col > 0.15 && col < 2.25) {
          c = vec3(0.3, 0.38, 0.45);
          if (col < 0.25 || col > 2.15 || y > 3.5 || y < 0.4 || abs(y - 2.0) < 0.05) c = vec3(0.7, 0.72, 0.74);
          if (y > 2.0 && col > 0.3 && col < 1.4 && y < 3.4) c = vec3(0.4, 0.5, 0.58);
        }
      } else {
        if (mod(y - 3.8, 2.6) < 0.1) c *= 0.84;
        if (mod(vUv2.x, 6.0) < 0.07) c *= 0.9;
      }
      if (y > top - 1.0 && y < top - 0.2) c = vec3(0.86, 0.85, 0.82);
      if (y > top - 0.2) c *= 0.8;
    } else if (type == 29) {
      // Mosque: light walls, tall narrow arched windows in rows, a decorative band under the eaves.
      float cx = mod(vUv2.x + 1.0, 2.0) - 1.0, y = vUv2.y, top = vAxis;
      float wy = 3.8;
      for (int r = 0; r < 2; r++) {
        float sy = y - (1.2 + float(r) * 4.6);
        bool arch = sy > 2.4 && length(vec2(cx * 1.0, sy - 2.4)) < 0.4;
        if (abs(cx) < 0.4 && sy > 0.0 && (sy <= 2.4 || arch) && y < top - 1.8) {
          c = vec3(0.2, 0.3, 0.42);
          if (abs(cx) > 0.31 || sy < 0.12 || length(vec2(cx, sy - 2.4)) > 0.31 && arch) c = vec3(0.8, 0.78, 0.7);
          if (abs(cx) < 0.04 || abs(sy - 1.4) < 0.04) c = vec3(0.72, 0.7, 0.62);
        }
      }
      if (y > top - 1.4 && y < top - 0.2) {
        c = vec3(0.7, 0.62, 0.45);
        float t = mod(vUv2.x, 0.8);
        if (t < 0.4) c = vec3(0.84, 0.8, 0.7);
        if (y > top - 0.5) c = vec3(0.9, 0.88, 0.8);
      }
      if (y < 0.5) c *= 0.8;
    } else if (type == 30) {
      // Bitterzoutloods (1919), now lofts: brick piers every 5.5 m with concrete caps, a brick relief band under the
      // coping, two tall narrow windows per bay upstairs, a loading dock and sliding barn doors downstairs.
      float bay = mod(vUv2.x, 5.5), y = vUv2.y, top = mod(vAxis, 100.0), d = top - y;
      float rel = bay - 0.7;
      vec3 concrete = vec3(0.8, 0.79, 0.75);
      bool pier = bay < 0.7;
      if (mod(y, 0.18) < 0.02) c *= 0.93;
      if (pier) c *= 1.1;
      if (d < 0.3) {
        c = concrete;
        if (!pier) c *= 0.93;
      } else if (d < 1.2) {
        if (!pier) {
          vec2 cell = floor(vec2(vUv2.x, y) / 0.25);
          c *= mod(cell.x + cell.y, 2.0) < 1.0 ? 0.78 : 1.12;
        } else if (d < 0.5) c = concrete;
      } else if (!pier) {
        if (d > 1.6 && d < 4.5) {
          // two windows per bay
          float wx = mod(rel - 0.8, 1.6);
          if (rel > 0.8 && rel < 4.0 && wx < 0.8) {
            c = vec3(0.17, 0.22, 0.28);
            if (wx < 0.07 || wx > 0.73 || d < 1.7 || d > 4.4 || abs(wx - 0.4) < 0.03 || abs(d - 3.0) < 0.03) c = vFrame;
          }
        } else if (y < 1.0) {
          c = concrete * 0.95;
          if (y < 0.12) c = vec3(0.2, 0.18, 0.17);
          else if (y > 0.85) c *= 1.06;
        }
        if (y >= 1.0 && y < 4.0) {
          if (rel > 0.35 && rel < 4.45) {
            c = vec3(0.22, 0.2, 0.19);
            if (rel < 3.15) {
              c = vec3(0.72, 0.73, 0.74);
              if (mod(y - 1.0, 0.5) < 0.04 || rel < 0.42 || abs(rel - 1.75) < 0.03) c = vec3(0.5, 0.51, 0.53);
            } else if (rel > 3.3 && rel < 4.3 && y > 1.05 && y < 3.6) {
              c = vFrame;
              if (rel > 3.45 && rel < 4.15 && y > 1.3 && y < 3.3) c = vec3(0.3, 0.34, 0.38);
            }
          }
          if (y > 3.9) c = vec3(0.12, 0.12, 0.13);
        }
      }
      if (pier && y < 1.0) c = concrete * 0.92;
      if (y < 0.05) c *= 0.6;
    } else if (type == 14) {
      // Brick chimney: a course line every few meters, a darker band below the top.
      if (mod(vUv2.y, 3.2) < 0.25) c *= 0.86;
      if (vUv2.y > vAxis - 2.5 && vUv2.y < vAxis - 1.0) c *= 0.7;
    } else if (type == 11) {
      // Lattice mast: steel cross-bracing.
      float a = fract((vUv2.x + vUv2.y) / 2.2), b = fract((vUv2.y - vUv2.x) / 2.2);
      c = (a < 0.16 || b < 0.16 || mod(vUv2.y, 2.2) < 0.25) ? vec3(0.78, 0.8, 0.82) : vec3(0.42, 0.44, 0.47);
    }
  } else if (roof) {
    if (type >= 8 && type != 10 && type != 15 && type != 18 && type < 21) {
      // Spires and caps: plain slate or metal.
    } else if (type == 15) {
      // Station roof: folded metal panels (the folds themselves come from the light); standing seams along them.
      vec2 dir = vec2(cos(vAxis), sin(vAxis));
      if (mod(dot(vUv2, dir), 1.5) < 0.1) c *= 0.92;
    } else if (type == 18) {
      // Shed roofs (sheddaken) in rows across the hall: the steep north-light glazing dark, the long slope the roof's
      // own colour, lit brighter toward its ridge.
      vec2 dir = vec2(cos(vAxis), sin(vAxis));
      float t = mod(dot(vUv2, dir), 7.0);
      if (t < 1.6) c = vec3(0.32, 0.36, 0.42);
      else c *= 0.82 + 0.3 * (t - 1.6) / 5.4;
    } else if (type == 30) {
      // Loods roof: zinc with standing seams, and long glass strips (skylights) along the building; on a flat roof the
      // strips are wide.
      float ang = floor(vAxis / 100.0) / 1000.0; // loods: the axis in milliradians, packed with the height
      vec2 dir = vec2(cos(ang), sin(ang));
      float across = dot(vUv2, vec2(-dir.y, dir.x)), along = dot(vUv2, dir);
      bool isFlat = vWorldNormal.y >= 0.97;
      float t = mod(across, isFlat ? 4.0 : 3.0);
      c *= 0.92;
      if (mod(across, 0.9) < 0.07) c *= 0.88;
      if (t > (isFlat ? 1.0 : 1.6) && t < (isFlat ? 3.2 : 2.6) && mod(along, 12.0) > 0.8) {
        c = vec3(0.55, 0.72, 0.78);
        if (mod(along, 1.5) < 0.1) c = vec3(0.4, 0.46, 0.5);
        if (t < (isFlat ? 1.15 : 1.7)) c = vec3(0.4, 0.46, 0.5);
      }
    } else if (type == 3) {
      // Venlo greenhouse roof: rows of glass ridges, lit on one side.
      vec2 dir = vec2(cos(vAxis), sin(vAxis));
      float across = dot(vUv2, vec2(-dir.y, dir.x));
      float t = mod(across, 3.2);
      c = t < 1.6 ? vec3(0.88, 0.92, 0.93) : vec3(0.7, 0.79, 0.82);
      if (t < 0.22) c = vec3(0.56, 0.6, 0.62);
      float along = dot(vUv2, dir);
      if (mod(along, 5.0) < 0.2) c *= 0.93;
    } else {
      // The real roof from the aerial photo, when there is one: solar panels, skylights, chimneys.
      if (vAtlas.x >= 0.0) c = texture2D(roofAtlas, vAtlas).rgb;
      if (vWorldNormal.y < 0.97) {
        // Pitched roof: courses of tiles along the contour.
        if (mod(vWorld.y, 0.55) < 0.16) c *= vAtlas.x >= 0.0 ? 0.93 : 0.86;
      } else if (vAtlas.x < 0.0) {
        // Flat roof without a photo: gravel speckle.
        float n = fract(sin(dot(floor(vUv2 * 2.0), vec2(12.9898, 78.233))) * 43758.5453);
        c *= 0.95 + 0.08 * n;
      }
    }
  }
  return c;
}
`;

async function makeBuildings() {
  const meta = await (await fetch(`${DATA}/buildings.json`)).json();
  const v = await fetchF32(`${DATA}/buildings.bin`);
  const COLS = 12; // x y z, normal, u v, kind, building, atlas u v (see scripts/build_buildings.py)
  const n = v.length / COLS;
  const atlasUv = new Float32Array(n * 2);
  const atlas = await new THREE.TextureLoader().loadAsync(`${DATA}/roof_atlas.png`);
  atlas.generateMipmaps = false;
  atlas.minFilter = THREE.LinearFilter;
  atlas.magFilter = THREE.LinearFilter;
  const pos = new Float32Array(n * 3), nor = new Float32Array(n * 3), uv = new Float32Array(n * 2);
  const kind = new Float32Array(n), type = new Float32Array(n), axis = new Float32Array(n), col = new Float32Array(n * 3);
  const frame = new Float32Array(n * 3);
  const walls = meta.map(wallColor), roofs = meta.map(roofColor), frames = meta.map(frameColor);
  for (let i = 0; i < n; i++) {
    const o = i * COLS;
    atlasUv.set([v[o + 10], v[o + 11]], i * 2);
    pos.set([v[o], v[o + 1], v[o + 2]], i * 3);
    nor.set([v[o + 3], v[o + 4], v[o + 5]], i * 3);
    uv.set([v[o + 6], v[o + 7]], i * 2);
    kind[i] = v[o + 8];
    const b = v[o + 9], m = meta[b];
    type[i] = TYPES[m.kind] ?? 0;
    axis[i] = m.kind === 'greenhouse' || m.kind === 'station' || m.kind === 'hall' ? m.axis : m.kind === 'factory' || m.kind === 'cinema' || m.kind === 'mosque' ? m.height
      // loods needs both: its axis (for the roof) in whole milliradians times 100, plus its height (under 100 m).
      : m.kind === 'loods' ? Math.round((((m.axis % Math.PI) + Math.PI) % Math.PI) * 1000) * 100 + Math.min(99, m.height) : (m.param ?? m.axis);
    frame.set([frames[b].r, frames[b].g, frames[b].b], i * 3);
    const c = kind[i] === 2 ? roofs[b] : walls[b];
    col.set([c.r, c.g, c.b], i * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setAttribute('aUv', new THREE.BufferAttribute(uv, 2));
  g.setAttribute('aKind', new THREE.BufferAttribute(kind, 1));
  g.setAttribute('aType', new THREE.BufferAttribute(type, 1));
  g.setAttribute('aAxis', new THREE.BufferAttribute(axis, 1));
  g.setAttribute('aFrame', new THREE.BufferAttribute(frame, 3));
  g.setAttribute('aAtlas', new THREE.BufferAttribute(atlasUv, 2));
  const mat = new THREE.MeshLambertMaterial({ vertexColors: true, flatShading: true, side: THREE.DoubleSide });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.roofAtlas = { value: atlas };
    sh.vertexShader = buildingVertex + sh.vertexShader.replace(
      '#include <begin_vertex>',
      `#include <begin_vertex>
       vUv2 = aUv; vKind = aKind; vType = aType; vAxis = aAxis; vFrame = aFrame; vAtlas = aAtlas;
       vWorld = position; vWorldNormal = normal;`,
    );
    sh.fragmentShader = buildingFragment + sh.fragmentShader.replace(
      '#include <color_fragment>',
      `#include <color_fragment>
       diffuseColor.rgb = buildingPattern(diffuseColor.rgb);`,
    );
  };
  return new THREE.Mesh(g, mat);
}

// ---------------------------------------------------------------------------------------------
// Trees

const LEAVES = [rgb(78, 124, 50), rgb(92, 138, 56), rgb(66, 112, 48), rgb(106, 146, 60), rgb(84, 116, 44), rgb(72, 130, 64)];
const NEEDLES = [rgb(44, 84, 54), rgb(52, 92, 58), rgb(38, 74, 50)];

function lumpySphere(seed) {
  const g = new THREE.IcosahedronGeometry(1, 2);
  const p = g.attributes.position;
  const seen = new Map();
  for (let i = 0; i < p.count; i++) {
    const key = `${p.getX(i).toFixed(3)},${p.getY(i).toFixed(3)},${p.getZ(i).toFixed(3)}`;
    if (!seen.has(key)) seen.set(key, 0.82 + 0.3 * hash(seen.size * 7.1 + seed * 31.7));
    const k = seen.get(key);
    p.setXYZ(i, p.getX(i) * k, p.getY(i) * k, p.getZ(i) * k);
  }
  // Share vertices so the normals come out smooth: a soft blob instead of a faceted gem.
  const smooth = mergeVertices(g.deleteAttribute('normal').deleteAttribute('uv'));
  smooth.computeVertexNormals();
  return smooth;
}

// Three flat tones per crown: lit, mid and shade, like a cel-shaded illustration.
function toonRamp() {
  const t = new THREE.DataTexture(new Uint8Array([70, 70, 70, 255, 150, 150, 150, 255, 255, 255, 255, 255]), 3, 1);
  t.minFilter = t.magFilter = THREE.NearestFilter;
  t.needsUpdate = true;
  return t;
}

async function makeTrees() {
  const t = await fetchF32(`${DATA}/trees.bin`);
  const n = t.length / 10;
  const group = new THREE.Group();
  const variants = [0, 1, 2].map(lumpySphere);
  const ramp = toonRamp();
  const leafMat = new THREE.MeshToonMaterial({ gradientMap: ramp });
  const needleMat = new THREE.MeshToonMaterial({ gradientMap: ramp });
  const trunkMat = new THREE.MeshLambertMaterial({ color: rgb(92, 72, 54) });
  const counts = [0, 0, 0];
  let conifers = 0;
  for (let i = 0; i < n; i++) {
    if (t[i * 10 + 5] > 0.5) conifers++;
    else counts[Math.floor(t[i * 10 + 9] * 3) % 3]++;
  }
  const crowns = variants.map((g, k) => new THREE.InstancedMesh(g, leafMat, counts[k]));
  const cones = new THREE.InstancedMesh(new THREE.ConeGeometry(1, 1, 7), needleMat, conifers);
  const trunks = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), trunkMat, n);
  const fill = [0, 0, 0];
  let ci = 0;
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < n; i++) {
    const [x, z, h, r, base, kind, , , , seed] = t.subarray(i * 10, i * 10 + 10);
    q.setFromAxisAngle(up, seed * Math.PI * 2);
    const trunkW = Math.min(0.7, 0.3 + h * 0.02);
    const trunkH = Math.max(1, Math.min(base, h - 2 * Math.max(1, (h - base) / 2, Math.min(r * 0.9, h * 0.42))) + 0.8);
    m.compose(p.set(x, trunkH / 2, z), q, s.set(trunkW, trunkH, trunkW));
    trunks.setMatrixAt(i, m);
    if (kind > 0.5) {
      const bottom = base * 0.5;
      m.compose(p.set(x, (bottom + h) / 2, z), q, s.set(r * 0.9, h - bottom, r * 0.9));
      cones.setMatrixAt(ci, m);
      cones.setColorAt(ci++, NEEDLES[Math.floor(seed * 97) % NEEDLES.length]);
    } else {
      const k = Math.floor(seed * 3) % 3;
      // Full, round crowns: seen from above, a tree is mostly crown.
      const half = Math.max(1, (h - base) / 2, Math.min(r * 0.9, h * 0.42));
      m.compose(p.set(x, h - half, z), q, s.set(r * 1.12, half, r * 1.12));
      crowns[k].setMatrixAt(fill[k], m);
      const c = LEAVES[Math.floor(seed * 1013) % LEAVES.length].clone();
      c.multiplyScalar(0.92 + 0.16 * hash(seed * 77));
      crowns[k].setColorAt(fill[k]++, c);
    }
  }
  for (const mesh of [...crowns, cones, trunks]) {
    mesh.frustumCulled = false;
    group.add(mesh);
  }
  return group;
}

// ---------------------------------------------------------------------------------------------
// Ground and diorama block


// ---------------------------------------------------------------------------------------------
// Sunken areas (landmarks.json "sunken"): ground and roads are cut out of them at y = 0 by a point-in-polygon test
// in the fragment shader, and drawn again `depth` lower, only inside, plus concrete walls around the edge.

const MAX_SUNKEN = 4, MAX_VERTS = 48;
const sunkenUniforms = {
  uSunkenPts: { value: Array.from({ length: MAX_SUNKEN * MAX_VERTS }, () => new THREE.Vector2()) },
  uSunkenN: { value: new Array(MAX_SUNKEN).fill(0) },
};
let sunken = []; // [{ depth, pts: [[x, z], ...] }] in local coordinates

async function loadSunken(area) {
  let list = [];
  try {
    const lm = await (await fetch(`${DATA}/landmarks.json`)).json();
    list = Array.isArray(lm.sunken) ? lm.sunken : [];
  } catch (e) { /* no landmarks in this era */ }
  sunken = list.slice(0, MAX_SUNKEN).map((s) => {
    let pts = s.area.map(([X, Y]) => [X - area.cx, area.cy - Y]);
    if (pts.length > 1 && pts[0][0] === pts.at(-1)[0] && pts[0][1] === pts.at(-1)[1]) pts = pts.slice(0, -1);
    return { depth: s.depth || 1.3, pts: pts.slice(0, MAX_VERTS) };
  }).filter((s) => s.pts.length >= 3);
  sunken.forEach((s, i) => {
    sunkenUniforms.uSunkenN.value[i] = s.pts.length;
    s.pts.forEach(([x, z], j) => sunkenUniforms.uSunkenPts.value[i * MAX_VERTS + j].set(x, z));
  });
}

const insideSunken = (x, z, s) => {
  let inside = false;
  for (let i = 0, j = s.pts.length - 1; i < s.pts.length; j = i++) {
    const [xi, zi] = s.pts[i], [xj, zj] = s.pts[j];
    if ((zi > z) !== (zj > z) && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
};

// Discards the fragments outside (inverse) or inside the sunken polygons. A no-op without any.
function patchSunken(mat, inverse) {
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = (sh, r) => {
    prev?.(sh, r);
    Object.assign(sh.uniforms, sunkenUniforms);
    sh.vertexShader = sh.vertexShader
      .replace('void main() {', 'varying vec2 vSunkenW;\nvoid main() {')
      .replace('#include <project_vertex>', '#include <project_vertex>\n  vSunkenW = (modelMatrix * vec4(transformed, 1.0)).xz;');
    sh.fragmentShader = sh.fragmentShader
      .replace('void main() {', `varying vec2 vSunkenW;
      uniform vec2 uSunkenPts[${MAX_SUNKEN * MAX_VERTS}];
      uniform int uSunkenN[${MAX_SUNKEN}];
      bool inSunken() {
        for (int k = 0; k < ${MAX_SUNKEN}; k++) {
          int n = uSunkenN[k];
          if (n < 3) continue;
          bool inside = false;
          for (int i = 0; i < ${MAX_VERTS}; i++) {
            if (i >= n) break;
            int jj = i == 0 ? n - 1 : i - 1;
            vec2 a = uSunkenPts[k * ${MAX_VERTS} + i], b = uSunkenPts[k * ${MAX_VERTS} + jj];
            if ((a.y > vSunkenW.y) != (b.y > vSunkenW.y) && vSunkenW.x < (b.x - a.x) * (vSunkenW.y - a.y) / (b.y - a.y) + a.x) inside = !inside;
          }
          if (inside) return true;
        }
        return false;
      }
      void main() {`)
      .replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>\n  if (${inverse ? '!' : ''}inSunken()) discard;`);
  };
  mat.customProgramCacheKey = () => (inverse ? 'sunken-in' : 'sunken-out');
  return mat;
}

// Duplicates every mesh of `group` that overlaps a sunken polygon, lowered by its depth and showing only the inside.
function addPits(group) {
  for (const s of sunken) {
    const xs = s.pts.map((p) => p[0]), zs = s.pts.map((p) => p[1]);
    const box = new THREE.Box3(new THREE.Vector3(Math.min(...xs), -1, Math.min(...zs)), new THREE.Vector3(Math.max(...xs), 1, Math.max(...zs)));
    for (const m of [...group.children]) {
      if (!m.userData.cut) continue;
      m.updateMatrixWorld(true);
      const b = new THREE.Box3().setFromObject(m);
      if (!b.intersectsBox(box)) continue;
      const mat = m.material.clone();
      mat.onBeforeCompile = undefined;
      patchSunken(mat, true);
      const pit = new THREE.Mesh(m.geometry, mat);
      pit.position.copy(m.position);
      pit.rotation.copy(m.rotation);
      pit.position.y -= s.depth;
      group.add(pit);
    }
  }
}

function makeSunkenWalls() {
  const group = new THREE.Group();
  const pos = [], nor = [], col = [];
  const WALL = [0.72, 0.71, 0.68], COPING = [0.55, 0.54, 0.51];
  for (const s of sunken) {
    let area2 = 0;
    s.pts.forEach(([x, z], i) => { const [x2, z2] = s.pts[(i + 1) % s.pts.length]; area2 += x * z2 - x2 * z; });
    // Pit-side normal of each edge: left of the direction for a polygon that runs one way, right for the other.
    const sgn = area2 > 0 ? 1 : -1;
    s.pts.forEach(([ax, az], i) => {
      const [bx, bz] = s.pts[(i + 1) % s.pts.length];
      const dx = bx - ax, dz = bz - az, len = Math.hypot(dx, dz);
      if (len < 1e-6) return;
      const nx = sgn * -dz / len, nz = sgn * dx / len;
      const quad = (y0, y1, c) => {
        let v = [[ax, y0, az], [bx, y0, bz], [bx, y1, bz], [ax, y1, az]];
        // Winding: (b - a) x up must point along the normal, otherwise flip.
        // (dx,0,dz) x (0,1,0) = (-dz, 0, dx)
        if (-dz * nx + dx * nz < 0) v = [v[1], v[0], v[3], v[2]];
        for (const k of [0, 1, 2, 0, 2, 3]) { pos.push(...v[k]); nor.push(nx, 0, nz); col.push(...c); }
      };
      quad(-s.depth, -0.14, WALL);
      quad(-0.14, 0.0, COPING);
    });
  }
  if (!pos.length) return group;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  group.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true })));
  return group;
}

async function makeGround(renderer, area) {
  const group = new THREE.Group();
  const loader = new THREE.TextureLoader();
  const size = area.chunk;
  const jobs = [];
  for (let ix = 0; ix < area.nx; ix++) {
    for (let iy = 0; iy < area.ny; iy++) {
      jobs.push(
        loader.loadAsync(`${DATA}/ground/${ix}_${iy}.png`).then((tex) => {
          tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
          tex.generateMipmaps = true;
          tex.minFilter = THREE.LinearMipmapLinearFilter;
          tex.magFilter = THREE.LinearFilter;
          tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
          const plane = new THREE.Mesh(new THREE.PlaneGeometry(size, size), patchSunken(new THREE.MeshBasicMaterial({ map: tex }), false));
          plane.userData.cut = true;
          plane.rotation.x = -Math.PI / 2;
          plane.position.set(-area.hx + size / 2 + ix * size, 0, area.hz - size / 2 - iy * size);
          group.add(plane);
        }),
      );
    }
  }
  await Promise.all(jobs);
  addPits(group);
  return group;
}

// Roads, bridges and paved squares as polygons on the ground, so their edges stay as sharp as a building's
// instead of blurring with the ground texture, plus a darker kerb line along their outlines. Every triangle and
// kerb segment goes to the 500 m chunks it touches, clipped to the chunk, sampling that chunk's road layer
// (scripts/build_ground.py).
const KERB = 0.72; // kerb brightness, as a fraction of the road surface next to it

async function makeRoads(renderer, area) {
  renderer.localClippingEnabled = true;
  const { polygons, edges } = await (await fetch(`${DATA}/roads.json`)).json();
  const size = area.chunk;
  const chunks = new Map();
  const chunk = (ix, iy) => {
    const key = ix * 1000 + iy;
    if (!chunks.has(key)) chunks.set(key, { ix, iy, tri: [], line: [] });
    return chunks.get(key);
  };
  // Calls fn(chunk) for every chunk the box [x0, x1] x [z0, z1] overlaps.
  const overlapping = (x0, x1, z0, z1, fn) => {
    const i0 = Math.max(0, Math.floor((x0 + area.hx) / size)), i1 = Math.min(area.nx - 1, Math.floor((x1 + area.hx) / size));
    const j0 = Math.max(0, Math.floor((area.hz - z1) / size)), j1 = Math.min(area.ny - 1, Math.floor((area.hz - z0) / size));
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) fn(chunk(i, j));
  };
  for (const { rings, bridge } of polygons) {
    // Without the closing point that repeats the first: triangulateShape drops it itself, in place, which would
    // shift the indices of every hole after it.
    const open = (r) => (r.length > 1 && r[0][0] === r.at(-1)[0] && r[0][1] === r.at(-1)[1] ? r.slice(0, -1) : r);
    const pts = rings.map((r) => open(r).map(([x, z]) => new THREE.Vector2(x, z)));
    const flat = pts.flat();
    const y = bridge ? 0.06 : 0.03;
    for (const [a, b, c] of THREE.ShapeUtils.triangulateShape(pts[0], pts.slice(1))) {
      const t = [flat[a], flat[b], flat[c]];
      const xs = t.map((p) => p.x), zs = t.map((p) => p.y);
      overlapping(Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs), (ch) => ch.tri.push(...t.map((p) => [p.x, y, p.y])));
    }
  }
  for (const ring of edges) {
    for (let k = 1; k < ring.length; k++) {
      const [ax, az] = ring[k - 1], [bx, bz] = ring[k];
      overlapping(Math.min(ax, bx), Math.max(ax, bx), Math.min(az, bz), Math.max(az, bz), (ch) => ch.line.push([ax, 0.08, az], [bx, 0.08, bz]));
    }
  }

  const loader = new THREE.TextureLoader();
  const group = new THREE.Group();
  await Promise.all([...chunks.values()].map(async ({ ix, iy, tri, line }) => {
    const cx0 = -area.hx + ix * size, cx1 = cx0 + size;
    const cz1 = area.hz - iy * size, cz0 = cz1 - size; // iy counts from the south edge; z points south
    const geometry = (verts) => {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(verts.flat(), 3));
      geo.setAttribute('uv', new THREE.Float32BufferAttribute(verts.flatMap(([x, , z]) => [(x - cx0) / size, (cz1 - z) / size]), 2));
      return geo;
    };
    const tex = await loader.loadAsync(`${DATA}/roads/${ix}_${iy}.png`);
    // Filtered like the ground, so the surface stays calm; the edges are sharp because the polygon ends there.
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    tex.generateMipmaps = true;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    const clippingPlanes = [
      new THREE.Plane(new THREE.Vector3(1, 0, 0), -cx0), new THREE.Plane(new THREE.Vector3(-1, 0, 0), cx1),
      new THREE.Plane(new THREE.Vector3(0, 0, 1), -cz0), new THREE.Plane(new THREE.Vector3(0, 0, -1), cz1),
    ];
    // Alpha ROAD_ALPHA (written as is, no blending) marks road pixels for the post pass, which keeps the
    // painterly filter off them so it can't round their edges.
    const common = { map: tex, clippingPlanes, blending: THREE.NoBlending, opacity: ROAD_ALPHA };
    if (tri.length) {
      // Triangulation can come out either way round; the camera only looks down at them, so draw both sides.
      { const m = new THREE.Mesh(geometry(tri), patchSunken(new THREE.MeshBasicMaterial({ ...common, side: THREE.DoubleSide }), false)); m.userData.cut = true; group.add(m); }
    }
    if (line.length) {
      { const m = new THREE.LineSegments(geometry(line), patchSunken(new THREE.LineBasicMaterial({ ...common, color: new THREE.Color(KERB, KERB, KERB) }), false)); m.userData.cut = true; group.add(m); }
    }
  }));
  addPits(group);
  return group;
}

function makeSlab(area) {
  // Peat over sand: the soil Helenaveen was dug out of.
  const c = document.createElement('canvas');
  c.width = 4;
  c.height = 280;
  const ctx = c.getContext('2d');
  const layers = [
    [0, 6, '#4b3a26'], [6, 10, '#2e2218'], [10, 60, '#3a2a1c'], [60, 64, '#2a1f16'], [64, 110, '#45321f'],
    [110, 116, '#2c2117'], [116, 150, '#56402a'], [150, 158, '#7a6a4a'], [158, 280, '#b39c6c'],
  ];
  for (const [a, b, col] of layers) {
    ctx.fillStyle = col;
    ctx.fillRect(0, a, 4, b - a);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  const side = new THREE.MeshLambertMaterial({ map: tex });
  const top = patchSunken(new THREE.MeshBasicMaterial({ color: 0x3a2a1c }), false); // cut open for the sunken areas
  const box = new THREE.Mesh(new THREE.BoxGeometry(area.hx * 2, SLAB, area.hz * 2), [side, side, top, top, side, side]);
  box.position.y = -SLAB / 2 - 0.02;
  return box;
}

// ---------------------------------------------------------------------------------------------
// Camera, lights and the outline pass

const deg = Math.PI / 180;

// look: the compass direction the camera looks in, in degrees; 45 is north-east, and every quarter turn works.
function makeCamera(scale, area, look) {
  const el = CAMERA_ELEVATION * deg, az = look * deg;
  // Screen x of a ground point is x cos(az) + z sin(az), its depth x sin(az) - z cos(az): at the corners of the area
  // that reaches hx |cos| + hz |sin| and hx |sin| + hz |cos|. Looking north-east, both are (hx + hz) / sqrt(2).
  const u = area.hx * Math.abs(Math.cos(az)) + area.hz * Math.abs(Math.sin(az));
  const vGround = (area.hx * Math.abs(Math.sin(az)) + area.hz * Math.abs(Math.cos(az))) * Math.sin(el);
  const width = Math.round(2 * u * scale);
  const top = vGround + TOP_MARGIN;
  const bottom = -vGround - SLAB * Math.cos(el) - 6;
  const height = Math.round((top - bottom) * scale);
  const cam = new THREE.OrthographicCamera(-width / scale / 2, width / scale / 2, top, top - height / scale, 10, 9000);
  const fwd = new THREE.Vector3(Math.cos(el) * Math.sin(az), -Math.sin(el), -Math.cos(el) * Math.cos(az));
  cam.position.copy(fwd.multiplyScalar(-4000));
  cam.up.set(0, 1, 0);
  cam.lookAt(0, 0, 0);
  return { cam, width, height };
}

function makeLights(scene) {
  const az = SUN_AZIMUTH * deg, el = SUN_ELEVATION * deg;
  // Toward the sun, in world axes: east = +x, north = -z.
  const dir = new THREE.Vector3(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el));
  const sun = new THREE.DirectionalLight(0xfff4e0, 0.95 * Math.PI);
  sun.position.copy(dir.multiplyScalar(1000));
  scene.add(sun);
  scene.add(new THREE.HemisphereLight(0xc8d8f0, 0x6a6450, 0.62 * Math.PI));
}

const outlineShader = {
  uniforms: { tColor: { value: null }, tDepth: { value: null }, texel: { value: new THREE.Vector2() }, range: { value: 1 }, bg: { value: BACKGROUND }, radius: { value: 0 },
    objectsPass: { value: false }, invViewProj: { value: new THREE.Matrix4() } },
  vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tColor; uniform sampler2D tDepth; uniform vec2 texel; uniform float range; uniform vec3 bg; uniform int radius;
    uniform bool objectsPass; uniform mat4 invViewProj;
    varying vec2 vUv;
    float depthAt(vec2 uv) { return texture2D(tDepth, uv).x * range; }

    // Kuwahara filter: each pixel takes the mean of whichever neighbouring quadrant is the most uniform.
    // It flattens grain into painted areas but keeps edges, the look of a hand-coloured illustration.
    // Road pixels (alpha below 0.75, see makeRoads) are skipped: a flat road would otherwise win every quadrant
    // next to it and spread over the verge.
    vec3 kuwahara(vec2 uv) {
      vec3 bestMean = texture2D(tColor, uv).rgb;
      float bestVar = 1e9;
      for (int q = 0; q < 4; q++) {
        vec2 dir = vec2(q == 1 || q == 3 ? 1.0 : -1.0, q >= 2 ? 1.0 : -1.0);
        vec3 m = vec3(0.0), s = vec3(0.0);
        float n = 0.0;
        for (int j = 0; j <= 4; j++) {
          for (int i = 0; i <= 4; i++) {
            if (i > radius || j > radius) continue;
            vec4 c = texture2D(tColor, uv + dir * vec2(float(i), float(j)) * texel);
            if (c.a < 0.75) continue;
            m += c.rgb; s += c.rgb * c.rgb; n += 1.0;
          }
        }
        if (n < 1.0) continue;
        m /= n;
        vec3 v = s / n - m * m;
        float var = v.r + v.g + v.b;
        if (var < bestVar) { bestVar = var; bestMean = m; }
      }
      return bestMean;
    }

    void main() {
      if (objectsPass) {
        // A mask of what stands up from the ground: white where the visible surface is more than 30 cm up (buildings,
        // trees, masts), black for the ground, roads and bridges. The viewer draws this part of the map again over
        // the cars and walkers that are behind it.
        float z = texture2D(tDepth, vUv).x;
        vec4 p = invViewProj * vec4(vUv * 2.0 - 1.0, z * 2.0 - 1.0, 1.0);
        gl_FragColor = vec4(vec3(z < 1.0 && p.y / p.w > 0.3 ? 1.0 : 0.0), 1.0);
        return;
      }
      vec4 c = texture2D(tColor, vUv);
      if (radius > 0 && c.a >= 0.75) c.rgb = kuwahara(vUv);
      float d = depthAt(vUv);
      float gap = 0.0;
      gap = max(gap, depthAt(vUv + vec2(texel.x, 0.0)) - d);
      gap = max(gap, depthAt(vUv - vec2(texel.x, 0.0)) - d);
      gap = max(gap, depthAt(vUv + vec2(0.0, texel.y)) - d);
      gap = max(gap, depthAt(vUv - vec2(0.0, texel.y)) - d);
      // A big step in depth means this pixel is the near side of a silhouette: ink it.
      if (gap > 2.2) c.rgb *= 0.58;
      gl_FragColor = vec4(c.rgb, 1.0);
    }`,
};

// ---------------------------------------------------------------------------------------------

async function main() {
  const params = new URLSearchParams(location.search);
  const scale = parseFloat(params.get('scale') || '3');
  const radius = parseInt(params.get('kuwahara') || '0');
  const objectsPass = params.get('pass') === 'objects';
  const look = parseFloat(params.get('look') || '45');
  const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setPixelRatio(1);
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = BACKGROUND;
  makeLights(scene);
  const area = await (await fetch(`${DATA}/area.json`)).json();
  area.hx = (area.xmax - area.xmin) / 2;
  area.hz = (area.ymax - area.ymin) / 2;
  await loadSunken(area);
  const [ground, roads, buildings, trees] = await Promise.all([makeGround(renderer, area), makeRoads(renderer, area), makeBuildings(), makeTrees()]);
  scene.add(ground, roads, buildings, trees, makeSlab(area), makeSunkenWalls());

  const { cam, width, height } = makeCamera(scale, area, look);
  let target = null;
  const post = new THREE.ShaderMaterial(outlineShader);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), post);
  const postScene = new THREE.Scene();
  postScene.add(quad);
  const postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  function ensureTarget(w, h) {
    if (target && target.width === w && target.height === h) return;
    target?.dispose();
    target = new THREE.WebGLRenderTarget(w, h, { depthBuffer: true });
    target.depthTexture = new THREE.DepthTexture(w, h, THREE.FloatType);
    target.texture.minFilter = target.texture.magFilter = THREE.NearestFilter;
  }

  const crop = document.createElement('canvas');
  window.renderTile = (tx, ty, tw, th) => {
    const m = 8; // margin in pixels, cropped off again below
    const x = tx - m, y = ty - m, w = tw + 2 * m, h = th + 2 * m;
    renderer.setSize(w, h, false);
    ensureTarget(w, h);
    cam.setViewOffset(width, height, x, y, w, h);
    cam.updateProjectionMatrix();
    renderer.setRenderTarget(target);
    renderer.render(scene, cam);
    post.uniforms.tColor.value = target.texture;
    post.uniforms.tDepth.value = target.depthTexture;
    post.uniforms.texel.value.set(1 / w, 1 / h);
    post.uniforms.range.value = cam.far - cam.near;
    post.uniforms.radius.value = radius;
    post.uniforms.objectsPass.value = objectsPass;
    post.uniforms.invViewProj.value.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse).invert();
    renderer.setRenderTarget(null);
    renderer.render(postScene, postCam);
    crop.width = tw;
    crop.height = th;
    crop.getContext('2d').drawImage(renderer.domElement, m, m, tw, th, 0, 0, tw, th);
    return crop.toDataURL('image/png');
  };
  window.sceneInfo = { width, height, scale, top: cam.top, look, area };
  window.ready = true;
}

main().catch((e) => {
  window.loadError = String(e.stack || e);
  console.error(e);
});
