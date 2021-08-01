#ifdef GL_ES
precision mediump float;
#endif

// The MIT License
// Copyright © 2013 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//------------------------------------------------------------------

float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }
float ndot( in vec2 a, in vec2 b ) { return a.x*b.x - a.y*b.y; }

float sdPlane( vec3 p )
{
    return p.y;
}

float sdSphere( vec3 p, float s )
{
    return length(p)-s;
}

float sdBox( vec3 p, vec3 b )
{
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float sdHollowBox( vec3 p, vec3 b, float e )
{
    p = abs(p  )-b;
    vec3 ep = vec3(e, 0.05, e);
  vec3 q = abs(p+ep)-ep;

  return min(min(
      length(max(vec3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
      length(max(vec3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
      length(max(vec3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}

float sdBoundingBox( vec3 p, vec3 b, float e )
{
       p = abs(p  )-b;
  vec3 q = abs(p+e)-e;

  return min(min(
      length(max(vec3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
      length(max(vec3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
      length(max(vec3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}

// vertical
float sdCylinder( vec3 p, vec2 h )
{
    vec2 d = abs(vec2(length(p.xz),p.y)) - h;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdLine( vec3 p, vec3 o, vec3 d, float l0, float l ) {
    // p: sdf pos to evaluate. o: ray origin. d: ray direction. l0: start length. l: ray length.
    vec3 dp = p - o;
    // closest point on ray
    float t = clamp(dot( dp, d ), l0, l);
    vec3 cp = o + t * d;
    // distance to closest point
    return length(p - cp);
}

//------------------------------------------------------------------
vec2 opU( vec2 d1, vec2 d2 )
{
    return (d1.x<d2.x) ? d1 : d2;
}

vec3 traj( int index, float time ) {
    // robot
    vec3 p = vec3(0.);
    float l = sin(time) / 2. + 0.5; // [0, 1], oscillating
    float l2 = cos(time) / 2. + 0.5; // [0, 1], oscillating (offset)
    if ( index == 0 ) { // robot
        vec3 p0 = vec3( 0.0,0.05,0.7);
        vec3 p1 = vec3( 0.0,0.05,-0.7);
        p = p0 * l2 + p1 * (1. - l2 );
    }
    if ( index == 1 ) { // human 1
        vec3 p0 = vec3( 0.4,0.15,-0.6);
        vec3 p1 = vec3(-0.6,0.15, 0.7);
        p = p0 * l + p1 * (1. - l );
    }
    if ( index == 2 ) { // human 1
        vec3 p0 = vec3(-0.4,0.15, 0.9);
        vec3 p1 = vec3( 0.6,0.15,-0.7);
        p = p0 * l + p1 * (1. - l );
    }
    if ( index == 3 ) { // human 1
        vec3 p0 = vec3(-0.4,0.15, 0.0);
        vec3 p1 = vec3( 0.0,0.15,-0.5);
        vec3 p2 = vec3( 0.4,0.15, 0.0);
        vec3 pa = p0 * l + p1 * (1. - l );
        vec3 pb = p1 * l + p2 * (1. - l );
        p = pa * l + pb * (1. - l );
    }
    return p;
}

//------------------------------------------------------------------
vec2 mapnorobot( in vec3 pos, in float time )
{
    vec2 res = vec2( 1e10, 0.0 );
    res = opU( res, vec2( sdCylinder(    pos-traj(1, time), vec2(0.05,0.15) ), 8.840 ) );
    res = opU( res, vec2( sdCylinder(    pos-traj(2, time), vec2(0.05,0.15) ), 8.840 ) );
    res = opU( res, vec2( sdCylinder(    pos-traj(3, time), vec2(0.05,0.15) ), 8.840 ) );
    // res = opU( res, vec2( sdBoundingBox( pos-vec3( 0.0,1., 0.0), vec3(1., 1., 1.), 0.005 ), 16.9 ) );
    res = opU( res, vec2( sdHollowBox(   pos-vec3( 0.5,0.1, 0.2), vec3(0.2,0.1,0.1), 0.005 ), 14.104 ) );
    res = opU( res, vec2( sdHollowBox(   pos-vec3(-0.5,0.1, -0.4), vec3(0.15,0.1,0.15), 0.005 ), 14.104 ) );
    //res = opU( res, vec2( sdCapsule(     pos-vec3( 1.0,0.00,-1.0),vec3(-0.1,0.1,-0.1), vec3(0.2,0.4,0.2), 0.1  ), 31.9 ) );
    res = opU( res, vec2( sdSphere(    pos-vec3(0.0,0.1, -0.7), 0.05 ), 16.9 ) );
    res = opU( res, vec2( sdHollowBox( pos, vec3(1., 0.1, 1.), 0.01 ), 16.9 ));

    return res;
}

vec2 map( in vec3 pos, in float time )
{
    vec2 res = vec2( 1e10, 0.0 );
    res = opU( res, vec2( sdCylinder(    pos-traj(0, time), vec2(0.05,0.05) ), 3.292 ) );
    res = opU( res, mapnorobot(pos, time));

    return res;
}


#define NLASERS 20
float[NLASERS] laserlengths;
void precompute_laser_lengths( in float time)
{
    float res = 1e10 ;
    const int N = NLASERS;
    for( int i=0; i<N; i++ )
    {
        float theta = float(i) / float(N) * 2. * 3.1416;
        vec3 dir = vec3(sin(theta), 0., cos(theta));
        vec3 orig = traj(0, time);
        // cast ray to find laser length
        float l = 0.06;
        float lmax = 2.;
        for( int j=0; j<24; j++ )
        {
            float h = mapnorobot( orig + dir*l , time ).x;
            //h = min(h, sdHollowBox( orig + dir*l, vec3(1., 0.1, 1.), 0.005 ));
            l += clamp( h, 0.02, 0.2 );
            if( h<0.004 || l>lmax ) break;
        }
        laserlengths[i] = l * 0.98;
    }
}

float lasermap( in vec3 pos, in float time )
{
    float res = 1e10 ;
    const int N = NLASERS;
    for( int i=0; i<N; i++ )
    {
        float theta = float(i) / float(N) * 2. * 3.1416;
        vec3 dir = vec3(sin(theta), 0., cos(theta));
        vec3 orig = traj(0, time);
        // cast ray to find laser length
        float l = laserlengths[i];
        res = min( res, sdLine(    pos, orig, dir, 0.06, l ) );
    }
    return res;
}

vec2 iBox( in vec3 ro, in vec3 rd, in vec3 rad )
{
    vec3 m = 1.0/rd;
    vec3 n = m*ro;
    vec3 k = abs(m)*rad;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    return vec2( max( max( t1.x, t1.y ), t1.z ),
                 min( min( t2.x, t2.y ), t2.z ) );
}

vec2 raycast( in vec3 ro, in vec3 rd, in float time )
{
    vec2 res = vec2(-1.0,-1.0);

    float tmin = 0.01;
    float tmax = 30.0;

    // raytrace floor plane
    float tp1 = (0.0-ro.y)/rd.y;
    if( tp1>0.0 )
    {
        tmax = min( tmax, tp1 );
        res = vec2( tp1, 1.0 );
    }
    //else return res;

    // raymarch primitives
    vec2 tb = vec2(tmin, tmax);//iBox( ro-vec3(0.0,0.4,-0.5), rd, vec3(10.,10.,10.) );
    if( tb.x<tb.y && tb.y>0.0 && tb.x<tmax)
    {
        //return vec2(tb.x,2.0);
        tmin = max(tb.x,tmin);
        tmax = min(tb.y,tmax);

        float t = tmin;
        precompute_laser_lengths(time);
        for( int i=0; i<70; i++ )
        {
            if (t > tmax) {break;}
            // detect hit
            vec2 h = map( ro+rd*t , time );
            if( abs(h.x)<(0.00001*t) )
            {
                res = vec2(t,h.y);
                break;
            }
            // detect laser
            float hl = lasermap( ro+rd*t, time );
            if( abs(hl)<(0.0015*t) )
            {
                res = vec2(t,4.8);
                break;
            }
            t += min(h.x, hl);
        }
    }
    return res;
}

// http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax , in float time)
{
    // bounding volume
    //float tp = (0.8-ro.y)/rd.y; if( tp>0.0 ) tmax = min( tmax, tp );

    float res = 1.0;
    float t = mint * 1.;
    for( int i=0; i<100; i++ )
    {
        float h = map( ro + rd*t , time ).x;
        float s = clamp(16.0*h/t,0.0,1.0);
        res = min( res, s*s*(3.0-2.0*s) );
        t += clamp( h, 0.02, 0.2 );
        if( res<0.004 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

// http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
vec3 calcNormal( in vec3 pos , in float time )
{
vec2 e = vec2(1.0,-1.0)*0.5773*0.0005;
    return normalize( e.xyy*map( pos + e.xyy, time ).x +
                      e.yyx*map( pos + e.yyx, time ).x +
                      e.yxy*map( pos + e.yxy, time ).x +
                      e.xxx*map( pos + e.xxx, time ).x );
}
float calcAO( in vec3 pos, in vec3 nor, in float time )
{
    float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float h = 0.01 + 0.12*float(i)/4.0;
        float d = map( pos + h*nor, time ).x;
        occ += (h-d)*sca;
        sca *= 0.95;
        if( occ>0.35 ) break;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 ) * (0.5+0.5*nor.y);
}

// http://iquilezles.org/www/articles/checkerfiltering/checkerfiltering.htm
float checkersGradBox( in vec2 p, in vec2 dpdx, in vec2 dpdy )
{
    // filter kernel
    vec2 w = abs(dpdx)+abs(dpdy) + 0.001;
    // analytical integral (box filter)
    vec2 i = 2.0*(abs(fract((p-0.5*w)*0.5)-0.5)-abs(fract((p+0.5*w)*0.5)-0.5))/w;
    // xor pattern
    return 0.5 - 0.5*i.x*i.y;
}

vec3 render( in vec3 ro, in vec3 rd, in vec3 rdx, in vec3 rdy, in float time )
{
    // background
    vec3 col = vec3(0.7, 0.7, 0.9) - max(rd.y,0.0)*0.3;

    // raycast scene
    vec2 res = raycast(ro,rd,time);
    float t = res.x;
    float m = res.y;
    if( m>-0.5 )
    {
        vec3 pos = ro + t*rd;
        vec3 nor = (m<1.5) ? vec3(0.0,1.0,0.0) : calcNormal( pos, time );
        vec3 ref = reflect( rd, nor );

        // material
        col = 0.2 + 0.2*sin( m*2.0 + vec3(0.0,1.0,2.0) );
        float ks = 1.0;

        if( m<1.5 )
        {
            // project pixel footprint into the plane
            vec3 dpdx = ro.y*(rd/rd.y-rdx/rdx.y);
            vec3 dpdy = ro.y*(rd/rd.y-rdy/rdy.y);

            float f = checkersGradBox( 3.0*pos.xz, 3.0*dpdx.xz, 3.0*dpdy.xz );
            //col = 0.15 + f*vec3(0.05);
            col = vec3(0., 0.4, 0.);  // ground color
            ks = 0.4;
        }

        if( m == 4.8 ) { // laser
            return vec3(1.0,0.,0.);
        }

        if ( true ) {
        // lighting
        float occ = calcAO( pos, nor, time );

        vec3 lin = vec3(0.0);

        // sun
        {
            vec3  lig = normalize( vec3(-0.5, 0.4, -0.6) );
            vec3  hal = normalize( lig-rd );
            float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
          //if( dif>0.0001 )
                  dif *= calcSoftshadow( pos, lig, 0.02, 4.5, time );
            float spe = pow( clamp( dot( nor, hal ), 0.0, 1.0 ),16.0);
                  spe *= dif;
                  spe *= 0.04+0.96*pow(clamp(1.0-dot(hal,lig),0.0,1.0),5.0);
            lin += col*2.20*dif*vec3(1.30,1.00,0.70);
            lin +=     5.00*spe*vec3(1.30,1.00,0.70)*ks;
        }
        // sky
        {
            float dif = sqrt(clamp( 0.5+0.5*nor.y, 0.0, 1.0 ));
                  dif *= occ;
            float spe = smoothstep( -0.2, 0.2, ref.y );
                  spe *= dif;
                  spe *= 0.04+0.96*pow(clamp(1.0+dot(nor,rd),0.0,1.0), 5.0 );
          //if( spe>0.001 )
                  spe *= calcSoftshadow( pos, ref, 0.02, 2.5, time );
            lin += col*0.60*dif*vec3(0.40,0.60,1.15);
            lin +=     2.00*spe*vec3(0.40,0.60,1.30)*ks;
        }
        // back
        {
            float dif = clamp( dot( nor, normalize(vec3(0.5,0.0,0.6))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
                  dif *= occ;
            lin += col*0.55*dif*vec3(0.25,0.25,0.25);
        }
        // sss
        {
            float dif = pow(clamp(1.0+dot(nor,rd),0.0,1.0),2.0);
                  dif *= occ;
            lin += col*0.25*dif*vec3(1.00,1.00,1.00);
        }

        col = lin;

        col = mix( col, vec3(0.7,0.7,0.9), 1.0-exp( -0.0001*t*t*t ) );
        }
    }

    return vec3( clamp(col,0.0,1.0) );
}

mat3 setCamera( in vec3 ro, in vec3 ta, float cr )
{
    vec3 cw = normalize(ta-ro);
    vec3 cp = vec3(sin(cr), cos(cr),0.0);
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv =          ( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

void shader( out vec4 fragColor, in vec2 fragCoord, in vec2 mo, in float time, in vec2 resolution )
{
    const int AA = 1;

    // camera
    float zoom = 1.5;
    vec3 ta = vec3( 0.0, -0., -0.0 );
    vec3 ro = ta + 1./zoom*vec3( 4.5*cos(0.1*time + 7.0*mo.x), 1.3 + 2.0*mo.y, 4.5*sin(0.1*time + 7.0*mo.x) );
    float s = sin(time * 0.5) / 2. + 0.5;
    float s4 = s*s*s*s;
    float phi = 1.5707 * s4 + 0.4 * (1.-s4);
    float theta = 1.5707 * s + 2.4 * (1.-s);
    float R = 6. * s4 + 3. * (1.-s4);
    ro = vec3(R*cos(phi)*cos(theta), R*sin(phi), 0.001+R*cos(phi)*sin(theta));
    // ro = vec3(0.1, 4., 0.1);
    // camera-to-world transformation
    mat3 ca = setCamera( ro, ta, 0.0 );

    vec3 tot = vec3(0.0);

    for( int m=0; m<AA; m++ )
    {
    for( int n=0; n<AA; n++ )
    {
        // pixel coordinates
        vec2 o = vec2(float(m),float(n)) / float(AA) - 0.5;
        vec2 p = (2.0*(fragCoord+o)-resolution.xy)/resolution.y;

        // focal length
        const float fl = 2.5;

        // ray direction
        vec3 rd = ca * normalize( vec3(p,fl) );

         // ray differentials
        vec2 px = (2.0*(fragCoord+vec2(1.0,0.0))-resolution.xy)/resolution.y;
        vec2 py = (2.0*(fragCoord+vec2(0.0,1.0))-resolution.xy)/resolution.y;
        vec3 rdx = ca * normalize( vec3(px,fl) );
        vec3 rdy = ca * normalize( vec3(py,fl) );

        // render
        vec3 col = render( ro, rd, rdx, rdy, time );

        // gain
        // col = col*3.0/(2.5+col);

        // gamma
        col = pow( col, vec3(0.4545) );

        tot += col;
    }
    }
    tot /= float(AA*AA);


    fragColor = vec4( tot, 1.0 );


}

// uncomment this line for shadertoy
//void mainImage( out vec4 fragColor, in vec2 fragCoord )
//{
//    shader(fragColor, fragCoord, iMouse.xy/iResolution.xy, iTime, iResolution.xy);
//}
// #if 0


// main function for editor.thebookofshaders.com / glslViewer
uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;
void main() {
    vec4 fragColor = vec4(0.);
    vec2 fragCoord = gl_FragCoord.xy;
    float time = u_time;
    vec2 mo = u_mouse.xy/u_resolution;
    vec2 resolution = u_resolution;

    shader( fragColor, fragCoord, mo, time, resolution );

    gl_FragColor = fragColor;
    vec2 st = gl_FragCoord.xy/u_resolution;
    //gl_FragColor = vec4(st.x,st.y,0.0,1.0);
}

// uncomment this line for shadertoy
// #endif
