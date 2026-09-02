// Shadow lookup against Panda's own shadow map.
//
// The board's shaders light themselves rather than using Panda's lighting, so
// they get no shadow for free the way fixed-function geometry does: the shadow
// map is only a depth texture, and a shader that never samples it will happily
// draw a lit fragment underneath a house.

// Only the fields actually used are declared. Panda binds struct members by
// name, so a subset is legal, and it keeps each shader honest about what it
// depends on. Index 0 is the directional light; ambient lights are not part of
// this array.
uniform struct p3d_LightSourceParameters {
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[1];

// 1.0 in full sun, 0.0 in full shadow. eyePos is the fragment in eye space,
// which is where Panda's shadowViewMatrix starts from -- not world space.
float sunShadow(vec4 eyePos) {
    vec4 sc = p3d_LightSource[0].shadowViewMatrix * eyePos;
    if (sc.w <= 0.0) {
        return 1.0;
    }

    // 3x3 PCF. A single tap resolves to the shadow map's own texels and reads
    // as a staircase, which on a flat mat is more distracting than no shadow.
    float s = 0.0;
    s += textureProjOffset(p3d_LightSource[0].shadowMap, sc, ivec2(-1, -1));
    s += textureProjOffset(p3d_LightSource[0].shadowMap, sc, ivec2( 0, -1));
    s += textureProjOffset(p3d_LightSource[0].shadowMap, sc, ivec2( 1, -1));
    s += textureProjOffset(p3d_LightSource[0].shadowMap, sc, ivec2(-1,  0));
    s += textureProj      (p3d_LightSource[0].shadowMap, sc);
    s += textureProjOffset(p3d_LightSource[0].shadowMap, sc, ivec2( 1,  0));
    s += textureProjOffset(p3d_LightSource[0].shadowMap, sc, ivec2(-1,  1));
    s += textureProjOffset(p3d_LightSource[0].shadowMap, sc, ivec2( 0,  1));
    s += textureProjOffset(p3d_LightSource[0].shadowMap, sc, ivec2( 1,  1));
    return s / 9.0;
}

// Shadow on a sunlit table is not an absence of light, it is skylight: cooler
// and dimmer, never black.
vec3 shadeSun(vec3 col, float lit) {
    return col * mix(vec3(0.55, 0.60, 0.74), vec3(1.0), lit);
}
