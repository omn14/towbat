#version 150

// Buildings and trees are flat-shaded coloured geometry: every face is one
// solid colour, which is what makes them read as untextured blocks next to the
// mat. This keeps the vertex colour as the base — the roof has to stay red and
// the timber has to stay pale — and layers the board's own material detail on
// top of it.

// 0 = building, 1 = foliage
uniform int modelKind;

in vec3 worldPos;
in vec3 worldNormal;
in vec4 vertColor;
in float localZ;
in vec4 eyePos;

out vec4 p3d_FragColor;

#pragma include "mat_noise.glsl"
#pragma include "shadow.glsl"

void main() {
    vec3 N = worldNormal;
    if (dot(N, N) < 0.0001) {
        N = vec3(0.0, 0.0, 1.0);
    }
    N = normalize(N);

    vec3 col = vertColor.rgb;
    // Triplanar-ish: use whichever pair of axes the face points away from, so
    // walls get vertical grain and roofs get horizontal, without UVs.
    vec3 an = abs(N);
    vec2 p = an.z > max(an.x, an.y) ? worldPos.xy
           : an.x > an.y            ? worldPos.yz
                                    : worldPos.xz;

    if (modelKind == 1) {
        // Foliage: clumps of leaf mass, plus a per-tree tonal shift so a wood
        // is not fifty identical cones. The shift is keyed on world position,
        // which differs per instance.
        col *= 0.86 + 0.28 * fbmG(worldPos.xy * 0.7);
        vec2 lp = p * 2.6;
        float lf = grainFade(lp);
        if (lf > 0.01) {
            col = mix(col, col * 1.35, specks(lp, 9.0, 0.55) * 0.45 * lf);
            col = mix(col, col * 0.62, specks(lp * 1.6, 21.0, 0.48) * 0.40 * lf);
        }
        col *= 1.0 + 0.10 * grainG(p * 9.0);
    } else {
        // Building: plaster and timber both weather, so break the flat fill
        // with broad staining and a fine grain rather than a pattern.
        col *= 0.90 + 0.20 * fbmG(p * 1.6);
        float stain = smoothstep(0.55, 0.88, fbmG(p * 3.2 + 7.7));
        col = mix(col, col * vec3(0.72, 0.70, 0.64), stain * 0.45);
        vec2 gp = p * 12.0;
        float gf = grainFade(gp);
        if (gf > 0.01) {
            col *= 1.0 + 0.14 * grainG(gp) * gf;
            // Sparse dark pitting: broken tiles, knots, missing daub.
            col = mix(col, col * 0.55, specks(gp * 0.7, 31.0, 0.30) * 0.30 * gf);
        }
    }

    // Ambient occlusion at the foot of the model. This is the part of a contact
    // shadow that can be done on the object itself, with no decal to sort or
    // depth-test against the ground.
    col *= mix(0.55, 1.0, clamp(localZ * 0.9, 0.0, 1.0));

    // Same key light and fill as terrain.frag, so models and ground agree.
    float lambert = clamp(dot(N, sunDir), 0.0, 1.0) * 0.7 + 0.3;
    col *= lambert;

    col = shadeSun(col, sunShadow(eyePos));

    p3d_FragColor = vec4(clamp(col, 0.0, 1.0), vertColor.a);
}
