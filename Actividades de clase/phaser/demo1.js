var w = 800;
var h = 400;
var jugador;
var fondo;

var bala, bala2, bala3, balaD = false;
var nave, nave2, nave3;

var salto, derecha, izquierda;
var menu;

var velocidadBala;
var despBala;

var estatusAire;
var estatusIzquierda = 0;

var nnNetwork, nnEntrenamiento, datosEntrenamiento = [];
var nnNetwork2, nnEntrenamiento2, datosEntrenamiento2 = []
var nnNetwork3, nnEntrenamiento3, datosEntrenamiento3 = []

var modoAuto = false, eCompleto = false;
var desplazo = false

const bala3_x = w - 185
const bala3_y = h - 405
const bala2_velocidad = 85

var juego = new Phaser.Game(w, h, Phaser.CANVAS, '', { preload: preload, create: create, update: update, render: render });

const resetBala2 = () => {
    bala2.x = w - 750;
    bala2.y = 5;
    bala2.body.velocity.y = bala2_velocidad
}

function preload() {
    juego.load.image('fondo', 'assets/game/db_escenario2-min.png');
    juego.load.spritesheet('mono', 'assets/sprites/kokun.png', 34, 48);
    juego.load.image('nave', 'assets/game/nave.png');
    juego.load.image('bala', 'assets/sprites/purple_ball.png');
    juego.load.image('menu', 'assets/game/menu.png');
    juego.load.image('goku_tieso', 'assets/game/koku_tieso.png', 38, 50);
}

function create() {
    juego.physics.startSystem(Phaser.Physics.ARCADE);
    juego.physics.arcade.gravity.y = 800;
    juego.time.desiredFps = 30;

    fondo = juego.add.tileSprite(0, 0, w, h, 'fondo');

    nave = juego.add.sprite(w - 110, h - 55, 'nave');
    nave2 = juego.add.sprite(w - 800, h - 400, 'nave');
    nave3 = juego.add.sprite(w - 200, h - 400, 'nave');

    
    jugador = juego.add.sprite(50, h, 'mono');
    juego.physics.enable(jugador);
    jugador.body.collideWorldBounds = true;
    jugador.animations.add('corre', [0, 1]);
    jugador.animations.play('corre', 6, true);

    bala = juego.add.sprite(w - 100, h, 'bala');
    juego.physics.enable(bala);
    bala.body.collideWorldBounds = true;

    bala2 = juego.add.sprite(w - 755, h - 405, 'bala');
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
    izquierda = juego.input.keyboard.addKey(Phaser.Keyboard.A);
    
    nnNetwork = new synaptic.Architect.Perceptron(2, 6, 6, 1);
    nnEntrenamiento = new synaptic.Trainer(nnNetwork);

    nnNetwork2 = new synaptic.Architect.Perceptron(3, 6, 6, 1);
    nnEntrenamiento2 = new synaptic.Trainer(nnNetwork2);

    nnNetwork3 = new synaptic.Architect.Perceptron(2, 6, 6, 1);
    nnEntrenamiento3 = new synaptic.Trainer(nnNetwork3);
}

const playerChangeSkin = (skin) => {
    if (skin == 'game over') {
        jugador.loadTexture('goku_tieso');
        return
    }
    jugador.loadTexture('mono');
    jugador.animations.add('corre', [0, 1]);
    jugador.animations.play('corre', 6, true);
}

const resetPositionBala3 = () => {
    bala3.position.x = bala3_x
    bala3.position.y = bala3_y
}

function enRedNeural() {
    nnEntrenamiento.train(datosEntrenamiento, { rate: 0.0003, iterations: 10000, shuffle: true });
    nnEntrenamiento2.train(datosEntrenamiento2, { rate: 0.0003, iterations: 10000, shuffle: true });
    nnEntrenamiento3.train(datosEntrenamiento3, { rate: 0.0003, iterations: 10000, shuffle: true });
}

function datosDeEntrenamiento(param_entrada) {
    nnSalida = nnNetwork.activate(param_entrada);
    const result = Math.round(nnSalida[0] * 100);
    return result >= 40;
}

function datosDeEntrenamiento2(param_entrada) {
    nnSalidaMov = nnNetwork2.activate(param_entrada);    
    const result = Math.round(nnSalidaMov[0] * 100);    
    return result > 20        
}

function datosDeEntrenamiento3(param_entrada) {
    nnSalidaMov3 = nnNetwork3.activate(param_entrada);
    const result = Math.round(nnSalidaMov3[0] * 100);    
    return result >= 9
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

                    enRedNeural();
                    eCompleto = true;
                    jugador.position.x = 50                    
                }
                modoAuto = true;
            }
            menu.destroy();
            resetVariables();            
            playerChangeSkin("initial")
            juego.paused = false;

        }
    }
}

function resetVariables() {
    jugador.body.velocity.x = 0;
    jugador.body.velocity.y = 0;
    bala.body.velocity.x = 0;
    bala.x = w - 100;
    balaD = false;
}

function saltar() {
    jugador.body.velocity.y = -270;
}

const desp_izquierda = () => {
    jugador.x = 0;    
    estatusIzquierda = 1;
    desplazo = true
}

function update() {
    fondo.tilePosition.x -= 1;
    bala2.body.velocity.y = bala2_velocidad    

    juego.physics.arcade.collide(bala, jugador, colisionH, null, this);
    juego.physics.arcade.collide(bala2, jugador, colisionH, null, this);
    juego.physics.arcade.collide(bala3, jugador, colisionH, null, this);

    bala3.body.velocity.y = 80
    bala3.body.position.x -= 5
    
    estatusAire = 0;
    estatusIzquierda = 0;

    if (!jugador.body.onFloor()) {        
        estatusAire = 1;
    }

    if (bala3.position.x <= 0 ) {
        resetPositionBala3()
    }
    
    despBala = Math.floor(jugador.x - bala.x);
    despBala2 = Math.floor(jugador.y - bala2.y);
    despBala3 = Math.floor(jugador.x - bala3.x);
    despBala3b = Math.floor(jugador.y - bala3.y);

    if(desplazo){
        if(bala.x > 600 && bala2.y < 280 && bala3.y < 280 ){
            desp_derecha();
        }                    
    }
    
    if (modoAuto == true) {
        juegoAutomatico()
    }

    if (balaD == false) {
        disparo();
    }

    if (bala.x <= 0) {
        resetVariables();
    }
    if (bala2.y >= 380) {
        resetBala2()
    }
    
    if (modoAuto == false) {
        juegoManual()
    }    
}

function juegoManual(){
    if (salto.isDown && jugador.body.onFloor()) {
        saltar();
    }

    if (izquierda.isDown) {
        desp_izquierda()
    }
    if (bala.x > 0) {
        datosEntrenamiento.push({
            'input': [despBala, velocidadBala],
            'output': [estatusAire]
        });

    }
    if (bala2.y > 100 && despBala2 > 0) {
        datosEntrenamiento2.push({
            'input': [despBala2, jugador.position.x, bala2.position.x],
            'output': [estatusIzquierda]
        });
    }
    if (bala3.y > 200 && bala3.x > 0) {
        datosEntrenamiento3.push({
            'input': [despBala3, despBala3b],
            'output': [ estatusIzquierda]
        });
    }
}

function juegoAutomatico(){
    if (bala2.y > 200) {
        const result = datosDeEntrenamiento2([despBala2, jugador.x, bala2.x])
        if (result) {            
            desp_izquierda()            
        }
    }

    if (bala3.y > 300 && bala3.x > 0) {
        if (datosDeEntrenamiento3([despBala3, despBala3b])) {
            desp_izquierda()            
        }
    }
    if (bala.x > 0 && jugador.body.onFloor()) {
        if (datosDeEntrenamiento([despBala, velocidadBala])) {
            saltar();
        }
    }    
}

function disparo() {
    const max = modoAuto ? 700 : 500
    velocidadBala = -1 * velocidadRandom(300, max);
    bala.body.velocity.y = 0;
    bala.body.velocity.x = velocidadBala;
    balaD = true;
}

function colisionH() {
    playerChangeSkin("game over")
    resetPositionBala3()
    resetBala2()
    jugador.x = 50
    pausa();
}

function velocidadRandom(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function render() {}