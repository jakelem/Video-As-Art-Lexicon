var CameraControls = require('3d-view-controls');
import {vec3, mat4} from 'gl-matrix';

class Camera {
  controls: any;
  projectionMatrix: mat4 = mat4.create();
  viewMatrix: mat4 = mat4.create();
  fovy: number = 45;
  aspectRatio: number = 1;
  near: number = 0.1;
  far: number = 1000;
  position: vec3 = vec3.create();
  direction: vec3 = vec3.create();
  target: vec3 = vec3.create();
  up: vec3 = vec3.create();
  time : number = 0;
  constructor(position: vec3, target: vec3) {
    this.controls = CameraControls(document.getElementById('canvas'), {
      eye: position,
      center: target,
     // rotateSpeed:0.0,
      //translateSpeed: 0.0,
     // zoomSpeed:0.0,
     // mode:'turntable'
    });

    vec3.add(this.target, this.position, this.direction);
    mat4.lookAt(this.viewMatrix, this.controls.eye, this.controls.center, this.controls.up);
  }

  animate() {
    //this.controls.tick();
    this.time ++;
    let d = vec3.fromValues(4.0 * Math.cos(0.005 * this.time), 0, 4.0 * Math.sin(0.005 * this.time))
    let ipos = vec3.fromValues(0,0,0);
    vec3.add(this.controls.eye,ipos,d);
    vec3.set(this.controls.center, 0.0,-0.3,0.0)
    mat4.lookAt(this.viewMatrix, this.controls.eye, this.controls.center, this.controls.up);

  }

  setAspectRatio(aspectRatio: number) {
    this.aspectRatio = aspectRatio;
  }

  updateProjectionMatrix() {
    mat4.perspective(this.projectionMatrix, this.fovy, this.aspectRatio, this.near, this.far);
  }

  update() {
    this.controls.tick();
    vec3.add(this.target, this.position, this.direction);
    mat4.lookAt(this.viewMatrix, this.controls.eye, this.controls.center, this.controls.up);
  }
};

export default Camera;
