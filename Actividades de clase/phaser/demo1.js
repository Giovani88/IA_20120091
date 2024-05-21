var w = 800;
var h = 400;
var jugador;
var fondo;

var bala, bala2, bala3, balaD = false, nave;

var salto, derecha, izquierda;
var menu;

var velocidadBala;
var despBala;



var estatusAire;
var estatuSuelo;

var nnNetwork, nnEntrenamiento, nnSalida, datosEntrenamiento = [];

var nnNetworkMov, nnEntrenamientoMov, datosEntrenamientoNN2 = []

var nnNetworkMov3, nnEntrenamientoMov3, datosEntrenamientoNN3 = []

var statusAireMov3 = 0;
var modoAuto = false, eCompleto = false;

var en_desplazo_derecha = false

var juego = new Phaser.Game(w, h, Phaser.CANVAS, '', { preload: preload, create: create, update: update, render: render });

const bala3_x = w - 90
const bala3_y = h - 400

var bloqueIA = false

var en_desplazo = false
var estatusDerecha = 0;
var estatusIzquierda = 0;
const resetBala2 = () => {
    bala2.body.velocity.y = -400;
    bala2.body.velocity.x = 0;
    bala2.position.x = w - 750;
    bala2.position.x = jugador.body.position.x;
    bala2.position.y = h - 800;
}

function preload() {
    juego.load.image('fondo', 'assets/game/fondo.jpg');
    juego.load.spritesheet('mono', 'assets/sprites/altair.png', 32, 48);
    juego.load.image('nave', 'assets/game/ufo.png');
    juego.load.image('bala', 'assets/sprites/purple_ball.png');
    juego.load.image('menu', 'assets/game/menu.png');

}

function create() {

    juego.physics.startSystem(Phaser.Physics.ARCADE);
    juego.physics.arcade.gravity.y = 800;
    juego.time.desiredFps = 30;

    fondo = juego.add.tileSprite(0, 0, w, h, 'fondo');
    nave = juego.add.sprite(w - 100, h - 70, 'nave');


    jugador = juego.add.sprite(50, h, 'mono');


    juego.physics.enable(jugador);
    jugador.body.collideWorldBounds = true;
    var corre = jugador.animations.add('corre', [8, 9, 10, 11]);
    jugador.animations.play('corre', 10, true);

    bala = juego.add.sprite(w - 100, h, 'bala');
    juego.physics.enable(bala);
    bala.body.collideWorldBounds = true;

    bala2 = juego.add.sprite(w - 750, h - 400, 'bala');
    juego.physics.enable(bala2);
    bala2.body.collideWorldBounds = false;

    bala3 = juego.add.sprite(bala3_x, bala3_y, 'bala');
    juego.physics.enable(bala3);
    bala3.body.collideWorldBounds = false;



    pausaL = juego.add.text(w - 100, 20, 'Pausa', { font: '20px Arial', fill: '#fff' });
    pausaL.inputEnabled = true;
    pausaL.events.onInputUp.add(pausa, self);
    juego.input.onDown.add(mPausa, self);

    salto = juego.input.keyboard.addKey(Phaser.Keyboard.SPACEBAR);

    derecha = juego.input.keyboard.addKey(Phaser.Keyboard.D);
    izquierda = juego.input.keyboard.addKey(Phaser.Keyboard.A);


    nnNetwork = new synaptic.Architect.Perceptron(2, 6, 6, 1);
    nnEntrenamiento = new synaptic.Trainer(nnNetwork);

    nnNetworkMov = new synaptic.Architect.Perceptron(3, 6, 6, 2);
    nnEntrenamientoMov = new synaptic.Trainer(nnNetworkMov);

    nnNetworkMov3 = new synaptic.Architect.Perceptron(3, 6, 6, 1);
    nnEntrenamientoMov3 = new synaptic.Trainer(nnNetworkMov3);

}

// function enRedNeural(){
//     nnEntrenamiento.train(datosEntrenamiento, {rate: 0.0003, iterations: 10000, shuffle: true});
// }

const resetPositionBala3 = () => {
    bala3.position.x = bala3_x
    bala3.position.y = bala3_y
}

function enRedNeural() {
    nnEntrenamiento.train(datosEntrenamiento, { rate: 0.0003, iterations: 10000, shuffle: true });
    nnEntrenamientoMov.train(datosEntrenamientoNN2, { rate: 0.0003, iterations: 10000, shuffle: true });
    nnEntrenamientoMov3.train(datosEntrenamientoNN3, { rate: 0.0003, iterations: 10000, shuffle: true });
}


function datosDeEntrenamiento(param_entrada) {

    //////console.log("Entrada",param_entrada[0]+" "+param_entrada[1]);
    ////console.log("Entrada",param_entrada[0]);
    nnSalida = nnNetwork.activate(param_entrada);
    ////console.log(nnSalida)
    var aire = Math.round(nnSalida[0] * 100);
    //var piso=Math.round( nnSalida[1]*100 );
    //////console.log("Valor ","En el Aire %: "+ aire + " En el suelo %: " + piso );
    ////console.log("Valor ","En el Aire %: "+ aire);
    return aire >= 40;
}

function datosDeEntrenamientoMov(param_entrada) {

    nnSalidaMov = nnNetworkMov.activate(param_entrada);

    var izquierda = Math.round(nnSalidaMov[0] * 100);
    // var derecha=Math.round( nnSalidaMov[1]*100 );

    console.log(nnSalidaMov)
    var derecha = Math.round(nnSalidaMov[1] * 100);
    ////console.log("izquierda,derecha",izquierda,derecha)
    if (nnSalidaMov[0] <= 0.055 && nnSalidaMov[1] <= 0.055) {
        //console.log('NO HACER NADA')
        return 0
    }
    else if (nnSalidaMov[1] > nnSalidaMov[0]) {
        //console.log('DEBO IR DE')
        return 2
    }
    else {
        //console.log('DEBO IR IZ')
        return 1
    }
    //return nnSalidaMov[1]>nnSalidaMov[0]
}

function datosDeEntrenamientoMov3(param_entrada) {

    nnSalidaMov3 = nnNetworkMov3.activate(param_entrada);
    const result = Math.round(nnSalidaMov3[0] * 100);    
    console.log(nnSalidaMov3, result)    
    return result > 80
}



function pausa() {
    juego.paused = true;
    menu = juego.add.sprite(w / 2, h / 2, 'menu');
    menu.anchor.setTo(0.5, 0.5);
}

function mPausa(event) {
    if (juego.paused) {
        var menu_x1 = w / 2 - 270 / 2, menu_x2 = w / 2 + 270 / 2,
            menu_y1 = h / 2 - 180 / 2, menu_y2 = h / 2 + 180 / 2;

        var mouse_x = event.x,
            mouse_y = event.y;

        if (mouse_x > menu_x1 && mouse_x < menu_x2 && mouse_y > menu_y1 && mouse_y < menu_y2) {
            if (mouse_x >= menu_x1 && mouse_x <= menu_x2 && mouse_y >= menu_y1 && mouse_y <= menu_y1 + 90) {
                eCompleto = false;
                datosEntrenamiento = [];
                modoAuto = false;
            } else if (mouse_x >= menu_x1 && mouse_x <= menu_x2 && mouse_y >= menu_y1 + 90 && mouse_y <= menu_y2) {
                if (!eCompleto) {
                    //////console.log("","Entrenamiento "+ datosEntrenamiento.length +" valores" );
                    enRedNeural();
                    eCompleto = true;
                }
                modoAuto = true;
            }

            menu.destroy();
            resetVariables();
            resetBala2()


            juego.paused = false;

        }
    }
}


function resetVariables() {
    jugador.body.velocity.x = 0;
    jugador.body.velocity.y = 0;
    bala.body.velocity.x = 0;
    bala.position.x = w - 100;
    //bala2.position.x= w-750;
    //bala2.position.y= h-800;
    //jugador.position.x=50;
    balaD = false;

}



function saltar() {
    jugador.body.velocity.y = -270;
}
const desp_derecha = () => {

    if (jugador.body.position.x > 100)
        return
    jugador.body.position.x = 50;
    estatusDerecha = 1;
    estatusIzquierda = 0;


}
const desp_izquierda = () => {

    jugador.body.position.x = 0;
    estatusDerecha = 0;
    estatusIzquierda = 1;
    en_desplazo = true

}



function update() {

    fondo.tilePosition.x -= 1;

    //juego.physics.arcade.collide(bala, jugador, colisionH, null, this);
    //juego.physics.arcade.collide(bala2, jugador, colisionH, null, this);
    //juego.physics.arcade.collide(bala3, jugador, colisionH, null, this);

    bala3.body.velocity.y = 250
    bala3.body.position.x -= 15

    

    estatuSuelo = 1;
    estatusAire = 0;

    estatusDerecha = 0;
    estatusIzquierda = 0;

    if (!jugador.body.onFloor()) {
        estatuSuelo = 0;
        estatusAire = 1;
        statusAireMov3=1;
    }else{
        statusAireMov3=0;
    }
    //console.log(bala3.position)
    if (bala3.position.x <= 0) {
        //console.log("REPONER BALA 3")
        resetPositionBala3()
    }

    // //console.log("jugador",jugador.body.position);
    // //console.log("bala2",bala2.body.position);
    despBala = Math.floor(jugador.position.x - bala.position.x);

    despBala2 = Math.floor(jugador.position.y - bala2.position.y);

    despBala3 = Math.floor(jugador.position.x - bala3.position.x);
    ////console.log(bala2.position.y)

    if (modoAuto == false && salto.isDown && jugador.body.onFloor()) {
        saltar();
    }

    if (modoAuto == false && derecha.isDown) {
        desp_derecha();
    }
    if (modoAuto == false && izquierda.isDown) {
        desp_izquierda()
    }

    if (modoAuto == true && bala2.position.y > 100) {
        const result = datosDeEntrenamientoMov([despBala2, jugador.position.x, bala2.position.x])

        if (!bloqueIA) {
            if (result == 2) { //derecha
                desp_derecha()
                console.log("ME MOVI derecha");
                bloqueIA = true;
            } else if (result == 1) {
                desp_izquierda()
                console.log("ME MOVI izquierda");
                bloqueIA = true;
            }
        }
    }
    // // if( modoAuto==true ){
    // //     desp_izquierda()
    // // }
    // if (modoAuto == true && bala.position.x > 0 && jugador.body.onFloor()) {
    //     if (datosDeEntrenamiento([despBala, velocidadBala])) {
    //         saltar();
    //     }
    // }
    if (modoAuto == true && bala3.position.x > 0 && jugador.body.onFloor()) {
        if (datosDeEntrenamientoMov3([despBala3, bala3.position.x,bala3.position.y])) {
            saltar();
            console.log("SALTA")
        }
    }

   
    en_desplazo = false

    bala2.body.velocity.y = 200
    if (bala2.body.position.y <= 0) {
        ////console.log("bola en el cielo")
        bloqueIA = false
    }


    if (balaD == false) {
        disparo();
    }

    if (bala.position.x <= 0) {
        resetVariables();
    }
    if (bala2.position.y >= 380) {
        resetBala2()
        // setTimeout(()=>{
        //     bala2.position.x= w-750;
        //     bala2.position.y= h-400;

        // },0)
    }


    // if (modoAuto == false && bala2.position.y > 100 && despBala2 > 0) {

    //     datosEntrenamientoNN2.push({
    //         'input': [despBala2, jugador.position.x, bala2.position.x],
    //         'output': [estatusIzquierda, estatusDerecha]
    //     });
    //     //console.log('input', despBala2, jugador.position.x, bala2.position.x)
    //     //console.log('output ', estatusIzquierda, estatusDerecha);
    // }

    // if (modoAuto == false && bala.position.x > 0) {

    //     datosEntrenamiento.push({
    //         'input': [despBala, velocidadBala],
    //         'output': [estatusAire]
    //     });

    //     // ////console.log("Desplazamiento Bala, Velocidad Bala, Estatus, Estatus: ",
    //     //     despBala + " " +velocidadBala + " "+ estatusAire+" "+  estatuSuelo);
    // }
    if(modoAuto == false && bala3.position.x > 0){
        datosEntrenamientoNN3.push({
            'input': [despBala3, bala3.position.x,bala3.position.y],
            'output': [statusAireMov3]
        });
        console.log('input',despBala3, bala3.position.x,bala3.position.y,'output',statusAireMov3)
    }
}

function disparo() {
    velocidadBala = -1 * velocidadRandom(300, 800);
    bala.body.velocity.y = 0;
    bala.body.velocity.x = velocidadBala;
    balaD = true;
}

function colisionH() {
    ////console.log('colission')

    pausa();
    //resetBala2()
}

function velocidadRandom(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function render() {

}
