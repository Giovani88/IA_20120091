var w = 600;
var h = 500;

var jugador;
var fondo;

var bola;
var menu;
var yAnterior;
var yActual;

const bola_x =0 
const bola_y =0
var nnNetwork, nnEntrenamiento, nnSalida, datosEntrenamiento = [];



var modoAuto = false, eCompleto = false;

var juego = new Phaser.Game(w, h, Phaser.CANVAS, '', { preload: preload, create: create, update: update, render: render});

var jugadorMoviendose = false
var salida_arriba = 0
var salida_izquierda = 0
var salida_derecha = 0
var salida_abajo = 0


function preload() {
    juego.load.image('bola',"assets/sprites/purple_ball.png")    
    juego.load.image('jugador', 'assets/sprites/goku.png');
    juego.load.image('menu', 'assets/game/menu.png');

}

function create() {
    //juego.physics.arcade.gravity.y = 800;

    juego.physics.startSystem(Phaser.Physics.ARCADE);

    jugador = juego.add.sprite(w /2 , h/2, 'jugador');
    juego.physics.enable(jugador);
    jugador.body.collideWorldBounds = true;
    jugador.body.gravity.y = 0
    jugador.body.immovable = true;

    bola = juego.add.sprite(bola_x,bola_y, 'bola');
    juego.physics.enable(bola);
    bola.body.bounce.set(1)
    bola.body.velocity.set(150)
    bola.body.collideWorldBounds = true;


    pausaL = juego.add.text(w - 100, 20, 'Pausa', { font: '20px Arial', fill: '#fff' });
    pausaL.inputEnabled = true;
    pausaL.events.onInputUp.add(pausa, self);
    juego.input.onDown.add(mPausa, self);

    

    izquierda = juego.input.keyboard.addKey(Phaser.Keyboard.A);
    derecha = juego.input.keyboard.addKey(Phaser.Keyboard.D);

    arriba = juego.input.keyboard.addKey(Phaser.Keyboard.W);
    abajo = juego.input.keyboard.addKey(Phaser.Keyboard.S);


    nnNetwork = new synaptic.Architect.Perceptron(5, 6, 6, 4);
    nnEntrenamiento = new synaptic.Trainer(nnNetwork);
}



function enRedNeural() {
    nnEntrenamiento.train(datosEntrenamiento, { rate: 0.0003, iterations: 10000, shuffle: true });
}

function datosDeEntrenamiento(param_entrada) {
    const salidas = nnNetwork.activate(param_entrada);
    const umbral = 0.030    
    const salidasValidas = salidas.filter((e)=>e>umbral);

    const value_max = Math.max(...salidasValidas)

    
    console.log(nnNetwork.toJSON())
    
    console.log(salidas)
    console.log(salidasValidas)
    console.log(value_max, salidas.indexOf(value_max))

    //var aire = Math.round(nnSalida[0] * 100);
    //return aire >= 40;
    return salidas.indexOf(value_max)
}


function pausa() {
    juego.paused = true;
    menu = juego.add.sprite(w / 2, h / 2, 'menu');
    menu.anchor.setTo(0.5, 0.5);
}

function entrenamiento(){    
console.log(JSON.stringify(datosEntrenamiento));
}

function resetVariables() {
    // jugador.body.velocity.x = 0;
    // jugador.body.velocity.y = 0;
    //jugador.body.gravity.y = 0
    jugador.x = w/2
    jugador.y = h/2
    bola.y = bola_y
    bola.x = bola_x
    // bola.body.velocity.x = 0;
    // bola.position.x = w - 100;
    // bolaD = false;
}

function desp_horizontal(direccion){
    if (jugadorMoviendose)
        return
    if (direccion == "A"){
        salida_izquierda = 1
    }else{
        salida_derecha = 1
    }
    //jugadorMoviendose = true;
    const x = (direccion == "A")? jugador.position.x - 200 : jugador.position.x + 200
    jugador.position.x = x
    setTimeout(()=>{
        jugadorMoviendose=false;
        jugador.position.x = w/2;
        salida_izquierda = salida_derecha = 0
    },1000)                            
}

function desp_vertical(direccion){
    if (jugadorMoviendose)
        return
    if (direccion == "W"){
        salida_arriba = 1
    }else{
        salida_abajo = 1
    }
    
    const y = (direccion == "W")? jugador.position.y - 200 : jugador.position.y + 200
    jugador.position.y = y
    setTimeout(()=>{
        jugadorMoviendose=false;
        jugador.position.y = h/2;
        salida_arriba = salida_abajo = 0
    },1000)                            
}

function update() {
    
    //juego.physics.arcade.collide(bola, jugador, colisionH, null, this);

    //Distancia obtenida con la formula de distancia euclidiana
    var distancia = Math.sqrt(Math.pow(bola.x - jugador.x, 2) + Math.pow(bola.y - jugador.y, 2));    
    //console.log(distancia)
    var distanciaBolaX = bola.x - jugador.x;
    var distanciaBolaY = bola.y - jugador.y;        


    // console.log(distancia)
    // if (distancia <= 140 ) {
    //     console.log("Los cuerpos van a colisionar.");
    // } 
    
    var cuadrante = getCuadrante(distanciaBolaX,distanciaBolaY)
    console.log(cuadrante)
    if(modoAuto == true && distancia <= 120){
        console.log("Rango para moverse")
        switch(datosDeEntrenamiento([bola.x,bola.y,distanciaBolaX,distanciaBolaY,distancia])){
            case 0: // arriba
                console.log("IA arriba")
                desp_vertical("W")
            break;
            case 1: //derecha
            console.log("IA derecha")

                desp_horizontal("D")
            break;
            case 2: //abajo
            console.log("IA abajo")

                desp_vertical("S")
            break;
            case 3: //izquierda
                console.log("IA izquierda")

                desp_horizontal("A")
            break
        }
        //console.log(datosDeEntrenamiento( [bola.x,bola.y,distancia, cuadrante]))
    }
    
    if(modoAuto == false && jugadorMoviendose){            
        datosEntrenamiento.push({
            'input': [bola.x,bola.y,distanciaBolaX,distanciaBolaY,distancia],
            'output': [salida_arriba,salida_derecha,salida_abajo,salida_izquierda]
        });
        console.log("input",bola.x,bola.y,distancia,distanciaBolaX,distanciaBolaY, " Output ", salida_arriba,salida_derecha,salida_abajo,salida_izquierda)
    }
    if(salida_abajo == 1 || salida_arriba == 1 || salida_derecha == 1 || salida_izquierda == 1){
        jugadorMoviendose = true;
    }

    if (modoAuto == false && izquierda.isDown) {        
        desp_horizontal("A");
    }
    if (modoAuto == false && derecha.isDown) {
        desp_horizontal("D");
    }
    if (modoAuto == false && arriba.isDown) {
        desp_vertical("W");
    }
    if (modoAuto == false && abajo.isDown) {
        desp_vertical("S");
    }
    
}

function getCuadrante(x, y) {
    if (x > 0 && y < 0) {
        return 1;
    } else if (x < 0 && y < 0) {
        return 2;
    } else if (x < 0 && y > 0) {
        return 3;
    } else if (x > 0 && y > 0) {
        return 4;
    }
    return 0
}

function colisionH() {
    console.log("colision")
    pausa();
}

function render() {
     // Mostrar el cuadro delimitador del jugador
     juego.debug.body(bola);
     juego.debug.body(jugador);
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
                    //jugador.position.x = 50
                    //resetPositionbola3()
                }
                modoAuto = true;
            }

            menu.destroy();
            resetVariables();
            //resetbola2()

            juego.paused = false;

        }
    }
}
