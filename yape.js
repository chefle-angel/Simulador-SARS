const express = require('express');
const cookieParser = require('cookie-parser');
const app = express();

app.set('trust proxy', true); 
app.use(cookieParser());

app.get('/', (req, res) => {
    const ip = req.headers['x-forwarded-for'] || req.socket.remoteAddress;
    
    console.log('--- Nueva Visita ---');
    console.log('IP:', ip);
    console.log('User-Agent:', req.headers['user-agent'] || 'Desconocido');
    console.log('Idioma:', req.headers['accept-language'] || 'Desconocido');
    console.log('Referrer:', req.headers['referer'] || 'Desconocido');
    console.log('Cookies:', req.cookies); 
    
    res.set('Cache-Control', 'no-store');
    res.send('Página cargada');
});

app.listen(process.env.PORT || 3000, () => console.log('Servidor activo'));
