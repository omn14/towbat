#version 150

// Draws the battle mat once, into a texture, at startup.
//
// This used to run per pixel per frame in c1.frag: five octaves of fbm, three
// of grain and a 3x3 cellular loop, to produce an image that never changes.
// With the board filling a 1920x1080 window that was 12 ms of a 27 ms frame,
// and it scaled with how far you zoomed in.

// The board rectangle in the card's UV space, so the table edge is baked in
// the same place the ground shader clips the grass to.
uniform vec2 boardMin;
uniform vec2 boardMax;

in vec2 texcoord;

out vec4 p3d_FragColor;

// Shared with terrain.frag so a hill is made of the same stuff as the mat.
#pragma include "mat_noise.glsl"

vec3 battleMat(vec2 uv) {
    vec2 p = uv * 16.0;                          // detail scale across the board
    vec3 col = matGrass(p);

    // Matt rather than poster green, but it is still grass: take only the
    // edge off the saturation.
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(lum), col, 0.95);

    return clamp(col, 0.0, 1.0);
}

void main() {
    // The target covers the board rectangle alone. The card it is drawn on is
    // half as much again in each direction, and texels spent on a margin that
    // gets discarded are what turned the grain to mush at 2048.
    vec2 uv = mix(boardMin, boardMax, texcoord);
    p3d_FragColor = vec4(battleMat(uv), 1.0);
}
