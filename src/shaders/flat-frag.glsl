#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;

in vec2 fs_Pos;
out vec4 out_Col;

float pi = 3.14159265359;
float degToRad = 3.14159265359 / 180.0;

struct MapQuery
{
  float dist;
  float displacement;
  float material;
};

struct Material
{
    vec3 color;
    float kd;
    float ks;
    float cosPow;
    float displacement;
};

struct PointLight
{
    vec3 position;
    vec3 color;
    bool castsShadow;
};

float hash3(vec3 v)
{
    return fract(sin(dot(v, vec3(24.51853, 4815.44774, 32555.33333))) * 3942185.3);
}

vec4 noise3(vec3 v)
{
    //Adapted from IQ: https://www.iquilezles.org/www/articles/morenoise/morenoise.htm
    vec3 intV = floor(v);
    vec3 fractV = fract(v);
    vec3 u = fractV*fractV*fractV*(fractV*(fractV*6.0-15.0)+10.0);
    vec3 du = 30.0*fractV*fractV*(fractV*(fractV-2.0)+1.0);
    
    float a = hash3( intV+vec3(0.f,0.f,0.f) );
    float b = hash3( intV+vec3(1.f,0.f,0.f) );
    float c = hash3( intV+vec3(0.f,1.f,0.f) );
    float d = hash3( intV+vec3(1.f,1.f,0.f) );
    float e = hash3( intV+vec3(0.f,0.f,1.f) );
    float f = hash3( intV+vec3(1.f,0.f,1.f) );
    float g = hash3( intV+vec3(0.f,1.f,1.f) );
    float h = hash3( intV+vec3(1.f,1.f,1.f) );
    
    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;
    
    
    vec3 dv = 2.0* du * vec3( k1 + k4*u.y + k6*u.z + k7*u.y*u.z,
                             k2 + k5*u.z + k4*u.x + k7*u.z*u.x,
                             k3 + k6*u.x + k5*u.y + k7*u.x*u.y);
    
    return vec4(-1.f+2.f*(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z), dv);
}

vec4 fbm3(vec3 v, int octaves, float amp, float freq, float pers, float freq_power)
{
    float sum = 0.f;
    vec3 dv = vec3(0.f,0.f,0.f);
    float speed = 0.01f;
    for(int i = 0; i < octaves; ++i)
    {
        amp *= pers;
        freq *= freq_power;
        vec4 noise = noise3((v) * freq);
        sum += amp * noise.x;
        dv += amp * noise.yzw;
    }
    return vec4(sum, dv);
}

MapQuery smoothMin( MapQuery a, MapQuery b, float k)
{
    float h = clamp( 0.5+0.5*(b.dist-a.dist)/k, 0.0, 1.0 );
    MapQuery res;
    if(h < 0.5) {
        res.dist = mix( b.dist, a.dist, h ) - k*h*(1.0-h);
        res.material = b.material;
    } else {
        res.dist = mix( b.dist, a.dist, h ) - k*h*(1.0-h);
        res.material = a.material;
    }
    return res;
}

vec2 smin( vec2 a, vec2 b, float k )
{
    float h = clamp( 0.5+0.5*(b.x-a.x)/k, 0.0, 1.0 );
    if(h < 0.5) {
        return vec2(mix( b.x, a.x, h ) - k*h*(1.0-h), b.y);
    } else {
        return vec2(mix( b.x, a.x, h ) - k*h*(1.0-h), a.y);
    }
}

vec2 smax( vec2 a, vec2 b, float k )
{
    float h = max(k-abs(a.x-b.x),0.0);
    return (a.x > b.x ? vec2((a.x + h*h*0.25/k), a.y) : vec2((b.x + h*h*0.25/k), b.y));
}


vec3 elongate( vec3 p, vec3 h )
{
    vec3 q = p - clamp( p, -h, h );
    return q;
}
float cappedCone(vec3 p, vec3 a, vec3 b, float ra, float rb)
{
    float rba  = rb-ra;
    float baba = dot(b-a,b-a);
    float papa = dot(p-a,p-a);
    float paba = dot(p-a,b-a)/baba;
    float x = sqrt( papa - paba*paba*baba );
    float cax = max(0.0,x-((paba<0.5)?ra:rb));
    float cay = abs(paba-0.5)-0.5;
    float k = rba*rba + baba;
    float f = clamp( (rba*(x-ra)+paba*baba)/k, 0.0, 1.0 );
    float cbx = x-ra - f*rba;
    float cby = paba - f;
    float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    return s*sqrt( min(cax*cax + cay*cay*baba,
                       cbx*cbx + cby*cby*baba) );
}

float roundCone(vec3 p, vec3 a, vec3 b, float r1, float r2)
{
    // sampling independent computations (only depend on shape)
    vec3  ba = b - a;
    float l2 = dot(ba,ba);
    float rr = r1 - r2;
    float a2 = l2 - rr*rr;
    float il2 = 1.0/l2;
    
    // sampling dependant computations
    vec3 pa = p - a;
    float y = dot(pa,ba);
    float z = y - l2;
    float x2 = dot( pa*l2 - ba*y, pa*l2 - ba*y );
    float y2 = y*y*l2;
    float z2 = z*z*l2;

    // single square root!
    float k = sign(rr)*rr*rr*x2;
    if( sign(z)*a2*z2 > k ) return  sqrt(x2 + z2)        *il2 - r2;
    if( sign(y)*a2*y2 < k ) return  sqrt(x2 + y2)        *il2 - r1;
                            return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}

// Ra: radius rb: roundedness h: height
float roundedCylinder( vec3 p, float ra, float rb, float h )
{
  vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}

float cappedTorus(vec3 p, vec2 sc, float ra, float rb)
{
  p.x = abs(p.x);
  float k = (sc.y*p.x>sc.x*p.y) ? dot(p.xy,sc) : length(p.xy);
  return sqrt( dot(p,p) + ra*ra - 2.0*ra*k ) - rb;
}

float box( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}


float sphere(vec3 p, float s)
{
    return length(p) - s;
}

float ellipsoid( vec3 p, vec3 r )
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

float stick(vec3 p, vec3 a, vec3 b, float r1, float r2)
{
    vec3 pa = p-a, ba = b-a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return  length( pa - ba*h ) - mix(r1,r2,h*h*(3.0-2.0*h));
}


vec3 bend( vec3 p, float k )
{
    float c = cos(k*p.x);
    float s = sin(k*p.x);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xy,p.z);
    return q;
}


vec3 twist( vec3 p, float k)
{
    float c = cos(k*p.y);
    float s = sin(k*p.y);
    mat2  m = mat2(c,-s,s,c);
    vec3  q = vec3(m*p.xz,p.y);
    return q;
}

float onion(float sdf, float thickness)
{
    return abs(sdf) - thickness;
}

float roundBox( vec3 p, vec3 b, float r )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

// https://iquilezles.org/www/articles/noacos/noacos.htm
mat3 rotationAxisAngle( vec3 v, float a )
{
    float si = sin( a );
    float co = cos( a );
    float ic = 1.0f - co;

    return mat3( v.x*v.x*ic + co,       v.y*v.x*ic - si*v.z,    v.z*v.x*ic + si*v.y,
                   v.x*v.y*ic + si*v.z,   v.y*v.y*ic + co,        v.z*v.y*ic - si*v.x,
                   v.x*v.z*ic - si*v.y,   v.y*v.z*ic + si*v.x,    v.z*v.z*ic + co );
}

float plane( vec3 p, vec3 n, float h )
{
  // n must be normalized
  return dot(p,n) + h;
}

float getBias(float time, float bias)
{
    return (time / ((((1.0/bias) - 2.0)*(1.0 - time))+1.0));
}

float getGain(float time,float gain)
{
  if(time < 0.5)
    return getBias(time * 2.0,gain)/2.0;
  else
    return getBias(time * 2.0 - 1.0,1.0 - gain)/2.0 + 0.5;
}



vec2 map(vec3 p)
{
//    if (abs(p.z) > 60.0 || abs(p.x) > 30.0 || abs(p.y) > 30.0)
//    {
//        return vec2(10000.0, 0.0);
//    }
    
    float modTime = mod(u_Time, 100.0 * pi);
    vec3 symP = vec3(abs(p.x), p.yz);
    //TODO: calculate these matrices on CPU or vert shader
    mat3 rot = rotationAxisAngle(normalize(vec3(1.0, 0.0, 0.0)), degToRad * 10.0);
    vec2 face = vec2(sphere(p + vec3(0.0, -0.8, 0.0), 1.2), 0.0);
    vec3 rotP = rot * (p - vec3(0.0,0.9,-0.63));
    face = smin(face, vec2(ellipsoid(p - vec3(0.0,0.0,-0.4), vec3(1.15, 1.3, 1.0)), 0.0), 0.2);
    // Forehead
    face = smin(face, vec2(roundBox(rotP, vec3(0.6, 0.2, 0.55), 0.2), 0.0), 0.6);
    // Chin
    face = smin(face, vec2(sphere(p + vec3(0.0, 0.98, 0.89), 0.33), 0.0), 0.3);
    // Eye socket
    face = smax(face, -vec2(sphere(symP + vec3(-0.5, -0.46, 1.2), 0.15), 0.0), 0.4);

    vec3 eyePos = vec3(-0.05, 0.03, 0.15);
    //eyelids
    //top eyelid
    vec3 blinkOffset = vec3(0.0, 0.1 + -0.2 * abs(sin(0.05 * modTime)), 0.0);
    //blinkOffset.y = 0.1;
    face = smin(face, vec2(stick(symP, vec3(0.4, 0.52, -1.3) + eyePos, vec3(0.75, 0.53, -1.28) + eyePos, 0.075, 0.075), 0.0), 0.07);
    
    face = smin(face, vec2(stick(symP, vec3(0.4, 0.52, -1.3) + eyePos + blinkOffset, vec3(0.7, 0.53, -1.28) + eyePos + blinkOffset, 0.075, 0.075), 0.0), 0.05);

    // low eyelid
    face = smin(face, vec2(stick(symP, vec3(0.76, 0.31, -1.15) + eyePos, vec3(0.3, 0.27, -1.37) + eyePos, 0.05, 0.05), 0.0), 0.08);
    face = smin(face, vec2(sphere(symP + vec3(-0.45, -0.04, 0.88), 0.36), 0.0), 0.15);

    // Eyeball
    vec3 eyeballPos = symP + vec3(-0.5, -0.44, 0.99);
    vec2 eyeball = vec2(sphere(eyeballPos, 0.19), 3.0);
    float pupilDist = length(eyeballPos.xy);
    // Iris
    if(pupilDist < 0.08 && pupilDist > 0.045) {
        eyeball.y = 6.0;
    }
    face = smin(face, eyeball, 0.0);
    
    // Cheekbones
    face = smin(face, vec2(stick(symP, vec3(0.92, 0.22, -0.64), vec3(0.52, -0.38, -0.84), 0.2, 0.25), 0.0), 0.24);
    // Hair Loop
    vec3 earOffset = vec3(-0.1, 0.0, 0.16);
    vec3 earP = elongate(symP + earOffset, vec3(0.05, 0.18, 0.1));
    earP.x += 0.01 * sin(10.0 * symP.y);
    earP.y += 0.02 * sin(5.0 * symP.y);

    vec2 ear = vec2(roundCone(earP, vec3(1.1, 0.24, -0.06), vec3(0.96, 0.04, -0.16), 0.15, 0.14), 0.0);
    face = smin(face, ear, 0.1);

    earP = elongate(symP + earOffset, vec3(0.01, 0.21, 0.06));
    earP.x += 0.01 * sin(20.0 * symP.y);
    earP.z += 0.026 * sin(30.0 * symP.x);
    
    vec2 earCut = vec2(-roundCone(earP, vec3(1.29, 0.26, -0.06), vec3(1.26, 0.15, -0.16), 0.14, 0.19), 0.0);
    face = smax(face, earCut, 0.1);

    
    vec3 bentSymP = bend(symP, -0.7);
    rot = rotationAxisAngle(normalize(vec3(0.8, 0.0, 1.0)), -degToRad * 60.0);
    
    vec3 rotSymP = rot * (symP + vec3(-1.7, -2.4, -0.3));
    rotSymP = bend(rotSymP, 0.05);
    rotSymP += 0.05 * sin(length(rotSymP * 4.0)) + 0.05 * cos(p * 3.5);

    vec2 hair = vec2(roundedCylinder(rotSymP, 1.1, 0.3, 0.1), 1.0);
    
    //scalp hair
    hair = smin(hair, vec2(sphere(symP + vec3(-0.3, -1.2, -0.3), 1.33), 1.0), 0.4);
    face = smin(face, hair, 0.1);
    vec3 twistPos = p;
    twistPos.xz += 0.03 * abs(sin(30.0 * p.y));
    vec2 neck = vec2(stick(twistPos, vec3(0.0, -0.4, 0.4), vec3(0.0, -2.8, 0.8), 0.8, 1.0), 2.0);
    face = smin(face, neck, 0.1);
    
    //lips
    vec3 lipPos = p + vec3(0.0, 0.49, 1.18);
    float xStretch = lipPos.x * 9.0;
    
    // Quadratic curve for lips
    float offset = -0.03 -0.08 * ((xStretch * xStretch * xStretch * xStretch) - 1.7 * (xStretch * xStretch));
    vec2 lip = vec2(roundBox(lipPos + vec3(0.0, -offset * 0.5, 0.0), vec3(0.16, 0.5 * offset + 0.01  * clamp(1.0 - (abs(lipPos.x * 200.0 * lipPos.x)), 0.0, 1.0), 0.1), 0.1), 0.0);
    
    //lipPos.y -= 0.01;
    lipPos = p + vec3(0.0, 0.52, 1.19);

    vec2 upperLip = vec2(roundBox(lipPos, vec3(0.2, 0.02, max(0.1 - abs(0.12 * p.x), 0.02)), 0.05), 0.0);
    lip = smin(lip, upperLip, 0.02);
    
    lipPos = p + vec3(0.0, 0.64, 1.14);
    float lipFunc = -10.0 * lipPos.x * lipPos.x + 0.4;
    if(lipFunc > lipPos.y) {
        lip.y = 4.0;
    }

    lipPos = p + vec3(0.0, 0.64, 1.14);
    lipFunc = 10.0 * lipPos.x * lipPos.x - 0.2;
    
    lipPos.y += 0.03 * (cos(13.0 * lipPos.x));
    
    vec2 lowerLip = vec2(roundBox(lipPos, vec3(0.13, 0.003, 0.1), 0.1), 0.0);
    
    if(lipFunc < lipPos.y) {
        lowerLip.y = 4.0;
    }
    
    lip = smin (lip, lowerLip, 0.002);
    face = smin(face, lip, 0.01);

    vec3 nosePos = vec3(0.0, -0.1, 0.0);
    vec2 nose = vec2(sphere(p + vec3(0.0,-0.02,1.55) - nosePos, 0.135), 0.0);
    
    // Nose bridge
    nose = smin(nose, vec2(stick(p, vec3(0.0, 0.5, -1.3) + nosePos, vec3(0.0, 0.14, -1.42) + nosePos, 0.11, 0.13), 0.0), 0.12);
    
    // Nose sides
    nose = smin(nose, vec2(stick(symP, vec3(0.1, 0.0, -1.55) + nosePos, vec3(0.13, -0.1, -1.45) + nosePos, 0.06, 0.06), 0.0), 0.09);
    face = smin(face, nose, 0.07);

    //body
    vec3 robeOffset = vec3(0.0, -0.4, -0.6);
    vec3 robeP = (p - vec3(0.0, -3.0, 0.0) + robeOffset);
    float fold = 0.1 * (getGain(0.5 * (sin(0.1 * modTime + p.x) + 1.0), 0.4));
    fold += 0.02 * sin(-p.y * 4.3 + p.x * 3.7);
    //fold += 0.01 * sin(0.5 + -p.y * 2.4 + symP.x * symP.x * 3.0);

    robeP.z += fold;
    robeP = elongate(robeP, vec3(1.0, 0.0, 0.0));
        
        
        
    vec2 robe = vec2(cappedCone(robeP, vec3(0.0, 2.6, 0.4), vec3(0.0, -3.4, -0.5), 2.4, 1.4), 5.0);
    if (robe.x < 0.0) {
        robe.y = 4.0;
    }
    
    vec3 foldP = p - vec3(0.0, -13.4, 0.9);
    foldP.z += 0.2 * (getGain(0.5 * (sin(0.1 * modTime + p.x) + 1.0), 0.4) - 0.5) * 2.0;

    vec2 bottomRobe = vec2(roundBox(foldP, vec3(1.4, 6.9, 0.7), 1.6), 5.0);
    
    bottomRobe.x += 0.001 * (0.5 * sin(p.x * 2.0) + sin(p.y * 1.0) + 2.0 * sin(p.z * 0.5));
    bottomRobe.x = onion(bottomRobe.x, abs(bottomRobe.x) * 0.1);
    bottomRobe = smin(bottomRobe, vec2(cappedCone(p, vec3(0.0, -16.6, 0.9), vec3(0.0, -22.9, -0.5), 2.1, 4.8), 5.0), 0.9);

    foldP = elongate(p, vec3(3.0, 0.0, 0.0));
    foldP += 0.2 * (sin(0.1 * modTime + p.x) + 1.0);
    //foldP.y += 0.3 * sin(0.1 * modTime + p.x);
    vec2 robeFolds = vec2(cappedCone(foldP, vec3(0.0, -18.9, 0.9), vec3(0.0, -22.4, 0.6), 0.4, 6.0), 5.0);
    vec2 redRobeFolds = vec2(cappedCone(foldP, vec3(0.0, -20.5, 0.1), vec3(0.0, -22.4, -0.0), 0.4, 6.3), 4.0);
    bottomRobe = smin(bottomRobe, robeFolds, 0.3);
    bottomRobe = smin(bottomRobe, redRobeFolds, 0.3);

    robe = smin(robe, bottomRobe, 0.1);
    vec2 insideRobe = robe.xy;

    //body
    vec2 body = smax(insideRobe, vec2(ellipsoid(p - vec3(0.0,-3.7,0.9), vec3(4.0, 1.4, 1.9)), 0.0), 0.01);
    face = smin(face, body, 0.1);
    robe.x = onion(robe.x, 0.03);
    
    vec2 belt = vec2(onion(robe.x, 0.1), 7.0);
    belt = smax(belt, vec2(roundBox(p - vec3(0.0, -4.5, 0.0), vec3(2.5, 0.9, 3.0), 0.01), 7.0), 0.01);
    robe = smin(robe, belt, 0.001);
    
    vec3 bicepFold = symP;
    bicepFold.z += 0.05 * sin(0.1 * p.x + 0.1 + 5.0 * symP.x - 2.5 * symP.y);
    bicepFold.z += 0.01 * sin(0.5 + 12.0 * symP.x - 5.0 * symP.y);

    vec3 longP = p - vec3(4.0, -6.0, -0.7);//elongate(p - vec3(4.0, -6.0, -0.7), vec3(0.0, 0.4, 0.0));
    vec2 bicep = vec2(stick(bicepFold, vec3(2.3, -2.6, 0.4), vec3(4.0, -5.6, -1.3), 0.7, 0.9), 5.0);
    bicep = smax(bicep, -insideRobe, 0.2);
    //robe = smin(robe, bicep, 0.2);
    vec3 sleeveP = symP - vec3(4.0, -6.0, -0.7);
    sleeveP += 0.03 * (sin(0.3 * modTime + p.x * 3.0 + p.y) + 1.0);
    sleeveP += 0.04 * sin(p.x * 2.7  + p.y * 3.0 + p.z * 1.0);
    vec2 sleeve = vec2(cappedCone(sleeveP, vec3(0.0, 0.0, -0.6), vec3(-4.4, -0.8, -4.0), 0.9, 1.8), 5.0);
    if (sleeve.x < 0.0) {
        sleeve.y = 4.0;
    }
    
    sleeve.x = onion(sleeve.x, 0.09);
    rot = rotationAxisAngle(vec3(0.0, 1.0, 0.0), -degToRad * 10.0);
    
    // Cut out sleeves using box
    rotP = symP - vec3(0.0, -4.0, -4.5);
    rotP.x += 0.05 * getBias((sin(0.3 * modTime + 4.0 * rotP.y) + 1.0) * 0.5, 0.6);
    
    rotP = rot * (rotP);
    sleeve = smax(sleeve, vec2(-roundBox(rotP, vec3(0.2, 5.0, 4.0), 0.1), 3.0), 0.1);
        
    longP = p - vec3(-4.0, -6.0, -0.7);
    
    vec2 lWrist = vec2(cappedCone(p - vec3(4.0, -5.2, -1.5), vec3(-3.0, -0.4, -1.7), vec3(-5.4, -0.4, -2.5), 0.3, 0.4), 0.0);
    vec2 rWrist = vec2(cappedCone(p - vec3(-4.0, -5.3, -1.6), vec3(3.0, -0.6, -1.8), vec3(5.4, -0.4, -1.5), 0.4, 0.4), 0.0);

    sleeve = smin(lWrist, sleeve, 0.01);
    sleeve = smin(rWrist, sleeve, 0.01);

    vec2 arm = smin(bicep, sleeve, 0.5);
    robe = smin(robe, arm, 0.01);
    // Cut off top of robe
    vec3 clothP = p - vec3(0.0, 2.45, 0.0);
    clothP.y += 0.03 * sin(0.3 * modTime + 4.0 * clothP.x);
    robe = smax(robe, vec2(-roundBox(clothP, vec3(4.0, 2.9, 4.5), 0.01), 0.0), 0.01);
    rot = rotationAxisAngle(vec3(0.0, 0.0, 1.0), degToRad *  45.0);
    rotP = rot * (p - vec3(0.0, 1.0, -1.2) + robeOffset);
        rotP += 0.02 * sin(0.3 + p.x * 2.0 + p.y * 3.2);
        rotP += 0.02 * sin(0.9 + p.x * 2.0 - p.y * 1.2);

    //Cut out vneck
    robe = smax(robe, vec2(-roundBox(rotP, vec3(3.5, 3.5, 1.0), 0.01), 0.0), 0.01);
    face = smin(face, robe, 0.01);
      /*
    vec2 testSphere = vec2(sphere(symP - vec3(2.0, 2.5, 0.0), 0.3), 0.0);
        face = smin(face, testSphere, 0.01);*/
        return face;

    
}

vec2 boundedMap(vec3 p)
{
    vec2 boundingBox = vec2(box(p - vec3(0.0,-7.4,0.0), vec3(9.5,16.0,9.0)), 0.0);
    if(boundingBox.x < 0.001) {
        return map(p);
    }
    return boundingBox;

}

vec3 calcNormals(vec3 p)
{
    float epsilon = 0.00001;
    return normalize(vec3(boundedMap(p + vec3(epsilon, 0.0, 0.0)).x - boundedMap(p - vec3(epsilon, 0.0, 0.0)).x, boundedMap(p + vec3(0.0, epsilon, 0.0)).x - boundedMap(p - vec3(0.0, epsilon, 0.0)).x, boundedMap(p + vec3(0.0, 0.0, epsilon)).x - boundedMap(p - vec3(0.0, 0.0, epsilon)).x));
    
}

vec4 raycast(vec3 origin, vec3 dir, int maxSteps)
{
    float t = 0.0;

    for(int i = 0; i < maxSteps; ++i)
    {
        vec3 p = origin + t * dir;
        vec2 dist = boundedMap(p);

        if (abs(dist.x) < 0.001) {
            return vec4(p, dist.y);
        }
        
        t += dist.x;
        if(t > 60.0) {
            return vec4(0.0, 0.0, 0.0, -100.0);

        }
    }
    
    return vec4(0.0, 0.0, 0.0, -100.0);
}

float softShadow(vec3 origin, vec3 dir, float minT, float maxT, float k)
{
    float res = 1.0;
    float ph = 1e20;

    for(float t = minT; t < maxT; )
    {
        vec3 p = origin + t * dir;
        vec2 dist = map(p);
        if (abs(dist.x) < 0.0001) {
            return 0.0;
        }
        /*
        float y = dist.x*dist.x/(2.0*ph);
        float d = sqrt(dist.x*dist.x-y*y);
        res = min( res, k*d/max(0.0,t-y) );
        ph = dist.x;*/
        res = min( res, k * dist.x / t );
        t += dist.x;

    }
    
    return res;

}

float flower(vec2 p, float r, float numPetals, float petalSize, float rotation)
{
    float angle = atan(p.y, p.x) + rotation;
    float rPetals = petalSize * (r + abs(cos(numPetals * angle)));
    return length(p) - rPetals;
}

float circle2d( vec2 p, float r )
{
    return length(p) - r;
}

float box2d( in vec2 p, in vec2 b )
{
    vec2 d = abs(p)-b;
    return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

float orientedBox2d( in vec2 p, in vec2 a, in vec2 b, float th )
{
    float l = length(b-a);
    vec2  d = (b-a)/l;
    vec2  q = (p-(a+b)*0.5);
          q = mat2(d.x,-d.y,d.y,d.x)*q;
          q = abs(q)-vec2(l,th)*0.5;
    return length(max(q,0.0)) + min(max(q.x,q.y),0.0);
}

Material robePattern(vec2 uv)
{
    Material matRes;
    float modTime = mod(u_Time, 100.0 * pi);
    vec4 fbm = fbm3(uv.xyy + vec3(0.0, 0.0, 0.0), 5, 1.0, 1.5, 0.4, 2.0);
   // fbm.x = 0.0;
    vec2 centerUv = uv - vec2(0.5);
    
    centerUv.x += 0.2 * sin(8.0 * centerUv.y);// + 0.7 * fbm.x;
    centerUv.x += 0.7 * fbm.x;
    centerUv.y += 0.2 * fbm.x;

    float growthTime = sin(0.04 * modTime);

    float res = box2d(centerUv, vec2(0.03, abs(growthTime) * 0.66));
    res = min(res, orientedBox2d(centerUv, vec2(0.0, 0.0), growthTime * vec2(0.2, 0.3), 0.02));
    res = min(res, orientedBox2d(centerUv, vec2(0.0, -0.1),growthTime *   vec2( -0.2, 0.3), 0.02));
    res = min(res, orientedBox2d(centerUv, vec2(0.0, -0.2), growthTime * vec2(  -0.6, 0.4), 0.04));
    res = min(res, orientedBox2d(centerUv, vec2(0.0, -0.4),  growthTime * vec2(-0.6, 0.1), 0.04));
    
    centerUv = uv - vec2(0.5);
    centerUv += 0.1 * fbm.x;
    float growthSign = sign(growthTime);
    float petalGrowth = abs(growthTime * 0.05);
    //res = min(res, circle2d(centerUv - vec2(0.0, 0.01), petalGrowth));
    res = min(res, flower(centerUv - vec2(-0.02 * growthSign, -0.0), 0.8, 3.0, petalGrowth * 1.3, growthTime));
    res = min(res, flower(centerUv - vec2(0.3 * growthSign, -0.2), 0.5, 2.5, petalGrowth * 0.8, growthTime));
    res = min(res, (res, flower(centerUv - vec2(0.1 * growthSign, 0.4), 0.8, 2.5, petalGrowth * 0.9, growthTime)));
    
    matRes.color = vec3(198.0,211.0,240.0) / 255.0;
    matRes.kd = 0.9;
    matRes.ks = 0.3;
    matRes.cosPow = 5.0;
    matRes.displacement = -0.8 * fbm.x;
    if (res < 0.0001) {
        matRes.kd = 1.0;
        matRes.ks = 1.0;
        matRes.cosPow = 40.0;
        matRes.displacement += -20.0 * abs(res);
        matRes.color =  mix(vec3(78.0,130.0,200.0) / 255.0, vec3(88.0,163.0,240.0) / 255.0, (uv.y));
    }
    
    float whiteFlowers = flower(centerUv - vec2(0.02, -0.1), 0.7, 3.0, petalGrowth, 2.0 * growthTime);
    whiteFlowers = min(whiteFlowers, (res, flower(centerUv - vec2(-0.2 * growthSign, 0.2), 0.8, 2.5, petalGrowth * 0.6, growthTime)));
    whiteFlowers = min(whiteFlowers, flower(centerUv - vec2(0.25 * growthSign, 0.3), 0.8, 3.0, petalGrowth * 1.1, growthTime));
    whiteFlowers = min(whiteFlowers, flower(centerUv - vec2(-0.0 * growthSign, -0.3), 0.8, 3.0, petalGrowth * 1.0, growthTime));

    if (whiteFlowers < 0.0001) {
        matRes.kd = 1.0;
        matRes.ks = 1.5;
        matRes.cosPow = 30.0;
        matRes.displacement += -20.0 * abs(whiteFlowers);
        matRes.color =  mix(vec3(200.0,240.0,250.0) / 255.0, vec3(230.0,240.0,250.0) / 255.0, (uv.y));
    }
    
    return matRes;
}

//https://stackoverflow.com/questions/34644101/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-produc
vec3 calcRobeNormals(vec2 uv)
{
    float epsilon = 0.01;
    Material left = robePattern(vec2(uv.x - epsilon, uv.y));
    Material right = robePattern(vec2(uv.x + epsilon, uv.y));
    Material up = robePattern(vec2(uv.x, uv.y - epsilon));
    Material down = robePattern(vec2(uv.x, uv.y + epsilon));
    
    float dzdx = (right.displacement - left.displacement) * 0.5;
    float dzdy = (up.displacement - down.displacement) * 0.5;
    
    return normalize(vec3(-dzdx, -dzdy, 1.0f));

}

Material beltPattern(vec2 uv)
{
    Material matRes;
    vec4 fbm = fbm3(uv.xyy + vec3(0.0, 0.0, 0.0), 5, 1.0, 1.5, 0.4, 2.0);

    float threshold = abs(cos(uv.x * 10.0));
    float dispThreshold = 2.0 * (threshold - 0.5);
    matRes.displacement = fbm.x * 3.0;
    if (threshold > 0.3) {
        matRes.ks = 1.0;
        matRes.kd = 1.0;
        matRes.cosPow = 20.0;
        matRes.color = vec3(0.62, 0.43, 0.6);
        matRes.displacement += (threshold - 0.3) * 10.0;

    } else {
        matRes.kd = 0.9;
        matRes.ks = 0.9;
        matRes.color = vec3(0.9, 0.54, 0.5);
        matRes.cosPow = 60.0;
        matRes.displacement += (0.3 - threshold) * 10.0;
    }
    matRes.displacement += -4.0 * getBias(abs(sin((1.2 * uv.x + uv.y) * 10.0)), 0.8);
    
    return matRes;
}

vec3 calcBeltNormals(vec2 uv)
{
    float epsilon = 0.002;
    Material left = beltPattern(vec2(uv.x - epsilon, uv.y));
    Material right = beltPattern(vec2(uv.x + epsilon, uv.y));
    Material up = beltPattern(vec2(uv.x, uv.y - epsilon));
    Material down = beltPattern(vec2(uv.x, uv.y + epsilon));
    
    float dzdx = (right.displacement - left.displacement) * 0.5;
    float dzdy = (up.displacement - down.displacement) * 0.5;
    
    return normalize(vec3(-dzdx, -dzdy, 1.0f));

}

Material hairPattern(vec2 uv)
{
    Material resMat;
    resMat.color = vec3(0.18, 0.16, 0.21);
    resMat.ks = 1.3;
    resMat.cosPow = 8.0;
    resMat.kd = 0.2;
    /*
    resMat.displacement = 0.4 * mod(uv.x * 7.0, 1.5);
    resMat.displacement += 0.3 * mod(pi + uv.x * 9.0, 1.9);
    resMat.displacement += 0.5 * mod(1.6 * pi + uv.x * 7.9, 2.1);
*/
    resMat.displacement = getBias(abs(0.3 * sin(uv.x * 26.0 + uv.y * 4.4)), 0.77);
    resMat.displacement += getBias(abs(0.2 * sin(pi * 0.9 + uv.x * 27.0 + uv.y * 4.3)), 0.8);
    resMat.displacement += 0.5 * sin(1.6 * pi + uv.x * 26.0 + uv.y * 4.3);

    return resMat;
    
}

vec3 calcHairNormals(vec2 uv)
{
    float epsilon = 0.01;
    Material left = hairPattern(vec2(uv.x - epsilon, uv.y));
    Material right = hairPattern(vec2(uv.x + epsilon, uv.y));
    Material up = hairPattern(vec2(uv.x, uv.y - epsilon));
    Material down = hairPattern(vec2(uv.x, uv.y + epsilon));
    
    float dzdx = (right.displacement - left.displacement) * 0.5;
    float dzdy = (up.displacement - down.displacement) * 0.5;
    
    return normalize(vec3(-dzdx, -dzdy, 1.0f));

}

Material backgroundPattern(vec2 uv)
{
    float modTime = mod(u_Time, 100.0 * pi);

    Material matRes;
    vec4 fbm = fbm3(uv.xyy + vec3(0.0, 0.0, 0.0), 5, 1.0, 1.5, 0.4, 2.0);
    vec2 centerUv = uv - vec2(0.5);
    
    centerUv.x += 0.2 * sin(8.0 * centerUv.y);// + 0.7 * fbm.x;
    centerUv.x += 0.7 * fbm.x;
    centerUv.y += 0.2 * fbm.x;

    float growthTime = sin(0.04 * modTime);
    float vines = box2d(centerUv, vec2(0.03, abs(growthTime) * 0.66));
    float a = getBias((uv.y + 1.0) * 0.5, 0.68);
    matRes.color = mix(vec3(0.58,0.6,0.72), vec3(0.92,0.95,0.96), a);
    if(vines < 0.001) {
        matRes.color = vec3(0.3, 0.01, 0.1);
        matRes.displacement += vines;
    }
    
    return matRes;
}

vec3 calcBackgroundNormals(vec2 uv)
{
    float epsilon = 0.01;
    Material left = backgroundPattern(vec2(uv.x - epsilon, uv.y));
    Material right = backgroundPattern(vec2(uv.x + epsilon, uv.y));
    Material up = backgroundPattern(vec2(uv.x, uv.y - epsilon));
    Material down = backgroundPattern(vec2(uv.x, uv.y + epsilon));
    
    float dzdx = (right.displacement - left.displacement) * 0.5;
    float dzdy = (up.displacement - down.displacement) * 0.5;
    
    return normalize(vec3(-dzdx, -dzdy, 1.0f));

}

void main() {
    float modTime = mod(u_Time, 100.0 * pi);

    float fov = 22.5f;
    float len = distance(u_Ref, u_Eye);
    vec3 look = normalize(u_Ref - u_Eye);
    vec3 right = normalize(cross(look, u_Up));
    float aspect = u_Dimensions.x / u_Dimensions.y;
    vec3 v = u_Up * len * tan(fov);
    vec3 h = right * len * aspect * tan(fov);

    vec3 p = u_Ref + fs_Pos.x * h + fs_Pos.y * v;
    vec3 dir = normalize(p - u_Eye);
    
    vec3 lightPos = vec3(1.0, 0.6, -20.0);
    
    PointLight[3] pointLights;
    pointLights[0] = PointLight(vec3(1.0, 20.0, -50.0), 0.9 * vec3(1.08,1.05,1.06), true);
    pointLights[1] = PointLight(vec3(-100.0, 0.6, 6.0), 0.75 * vec3(0.2,0.3,0.38), false);
    pointLights[2] = PointLight(vec3(80.0, 0.6, 4.0), 0.77 * vec3(0.2,0.3,0.4), false);

    vec4 isect = raycast(u_Eye, dir, 128);
    float a = getBias((fs_Pos.y + 1.0) * 0.5, 0.68);
    
    // Clear color
    vec3 albedo = mix(vec3(0.58,0.6,0.72), vec3(0.92,0.95,0.96), a);
    
    /*vec2 pos_uv = fs_Pos;
    pos_uv.y *= aspect;
    Material m = backgroundPattern(pos_uv);
    if(flower(fs_Pos.xy, 0.5, 3.0, 0.3, 1.0) < 0.001) {
    }
    albedo = m.color;

    vec3 backgroundNormal = calcBackgroundNormals(fs_Pos);
    vec3 lightBackground = normalize(vec3(1.0, 0.0, 1.0));
    float diffuseBackground = clamp(dot(backgroundNormal, lightBackground), 0.0, 1.0);
    vec3 col = diffuseBackground* albedo  ;*/
    
    vec3 col = albedo;
    if(isect.w >= 0.0)
    {
        col = vec3(0.0);
        vec3 normal = calcNormals(isect.xyz);
        vec3 tangent = normalize(cross(vec3(0,1,0),normal));
        vec3 bitangent = normalize(cross(normal, tangent));

        mat3 tbn = mat3(tangent, bitangent, normal);
        vec3 viewVec = normalize(isect.xyz - u_Eye.xyz);
        float kd = 1.0;
        float ks = 1.0;
        float cosPow = 28.0;

        if(isect.w == 0.0)
        {
            // Skin
            ks = 0.2;
            
            float skinLerp = clamp(abs(isect.z * 1.2) - 0.9, 0.0, 1.0);
            skinLerp = getBias(skinLerp, 0.6);
            
            albedo = vec3(0.7, 0.62, 0.67);

            vec3 cheeks = vec3(0.6, -0.19, -0.74);
            vec3 symIsect = vec3(abs(isect.x), isect.yz);
            float cheekDist = distance(cheeks, symIsect);
            cheekDist = clamp(cheekDist * 1.3, 0.0, 1.0);
            albedo = mix(vec3(0.72, 0.42, 0.62), albedo, cheekDist);

            cheeks = vec3(1.0, -0.36, -0.7);
            cheekDist = clamp(distance(cheeks, symIsect) * 1.5,0.0,1.0);
            albedo = mix(vec3(0.36, 0.21, 0.31), albedo, cheekDist);
            
            vec3 temples = vec3(0.85, 1.1, -1.09);
            float templeDist = clamp(distance(temples, symIsect) * 1.5, 0.0, 1.0);
            albedo = mix(vec3(0.3, 0.19, 0.4), albedo, templeDist);

            vec3 chest = vec3(0.0, -1.8, -0.6);
            float chestDist = clamp(distance(chest, symIsect) * 0.8, 0.0, 1.0);
            albedo = mix(vec3(0.6, 0.33, 0.67), albedo, chestDist);
            chestDist = clamp(distance(chest - vec3(0.0, 2.6, 0.0), symIsect) * 0.8, 0.0, 1.0);
            albedo = mix(vec3(0.6, 0.33, 0.67), albedo, chestDist);

            cosPow = 60.0;
        } else if (isect.w <= 1.0) {
            //hair
            vec3 symIsect = vec3(abs(isect.x), isect.yz);

            vec3 hairCenter = vec3(1.8, 2.5, 0.0);
            float hairDist = clamp(distance(hairCenter, symIsect) * 0.7, 0.0, 1.0);
            albedo = mix(vec3(0.8, 0.0, 0.0), vec3(0.0,0.8,0.0), hairDist);

            vec3 hairVector = symIsect.xyz - hairCenter;
            vec3 xAxis = normalize(-vec3(0.69512, 0.67625, 0.2439));
            xAxis = normalize(-vec3(1.0, 0.0, 0.0));

            float angle = acos(dot(normalize(hairVector.xy), xAxis.xy));
            float d = length(hairVector);
            
            albedo = vec3(mod(angle * 12.0, 2.0));
            vec2 uv = vec2(angle, d);
            Material hairMat = hairPattern(vec2(uv));
            albedo = hairMat.color;
            normal = tbn * calcHairNormals(uv);
            cosPow = hairMat.cosPow;
            kd = hairMat.kd;
            ks = hairMat.ks;
            //albedo = normal;
            //albedo = vec3(hairMat.displacement);
        } else if (isect.w <= 2.0) {
            // Neck
            vec4 fbm = fbm3(isect.xyz, 3, 1.0, 9.1, 0.5, 2.4);

            kd = 0.1;
            ks = 0.9 + 0.6 * (fbm.x + 1.0);
            cosPow = 12.0;
            albedo = vec3(0.7, 0.57, 0.1);
        } else if (isect.w <= 3.0) {
            // Eyes
            albedo = vec3(0.2, 0.2, 0.2);

        } else if (isect.w <= 4.0) {
            // Lips , red dress
            vec4 fbm = fbm3(isect.xyz, 3, 1.0, 1.1, 0.4, 2.0);

            cosPow = 3.0;
            
            ks = 0.5*(fbm.x + 1.0);
            ks = getBias(ks, 0.4) * 2.6;

            kd = 0.4;
            albedo = vec3(183.0, 24.0, 42.0) / 255.0;

        } else if (isect.w <= 5.0) {
            // dress, eyes
            kd = 0.9;
            ks = 1.2;
            albedo = vec3(226.0,234.0,236.0) / 255.0;
            float freqTime = abs(sin(modTime * 0.01)) * 0.5 + 1.9;
            
            vec3 uv = isect.xyz;
            //uv.x *= 1.2;
            uv.y *= 0.3;
            uv.y = fract(uv.y);
            //uv.x = fract(0.5 * uv.x);
            //uv.z = fract(0.3  * uv.z);

            //uv.x += sin(uv.y);
            float angle = acos(dot(normalize(uv), vec3(1.0,0.0,0.0)));
            angle += pi;
            vec2 uvAngle = vec2(angle, uv.y);
            uvAngle.x = fract(-0.02 + 0.79 * uvAngle.x);
            Material dressMat = robePattern(uvAngle);
            vec3 patternNorm = calcRobeNormals(uvAngle);
            normal = tbn * patternNorm;
            albedo = dressMat.color;
            ks = dressMat.ks;
            kd = dressMat.kd;
            cosPow = dressMat.cosPow;

        } else if (isect.w <= 6.0) {
            //  eyes
            albedo = vec3(177.0,180.0,210.0) / 255.0;

        } else if (isect.w <= 7.0) {
            //  belt
            vec3 center = floor(vec3(0.0, -4.3, 0.0));
            vec3 uv = isect.xyz - center;
            float angle = acos(dot(normalize(uv), vec3(1.0,0.0,0.0)));

            vec2 uvAngle = vec2(angle, uv.y);
            Material mat = beltPattern(uvAngle);
            vec3 beltNorm = calcBeltNormals(uvAngle);
            normal = tbn * beltNorm;

            ks = mat.ks;
            kd = mat.kd;
            albedo = mat.color;
            cosPow = mat.cosPow;
        } else {
            albedo = vec3(0.0);
        }

        for (int i = 0; i < 3; ++i) {
            vec3 lightVec = normalize(pointLights[i].position - isect.xyz);
            vec3 h = normalize(lightVec - viewVec);
            float diffuse = clamp(dot(normal, lightVec), 0.0, 1.0);
            float specularIntensity = max(pow(max(dot(h, normal), 0.f), cosPow), 0.f);
            
            float shadow = 1.0;
            if (pointLights[i].castsShadow) {
                shadow = softShadow(isect.xyz + normal * 0.04, lightVec, 0.02, 4.5, 32.0);
            }
            
            vec3 lightIntensity = shadow * pointLights[i].color * clamp(kd * diffuse + ks * specularIntensity, 0.0, 2.7);
            col += lightIntensity * albedo;
        }
        
        // Diffuse Light
        col += vec3(0.20, 0.21, 0.23) * albedo;
    }
        
    out_Col = vec4(col, 1.0);
}
