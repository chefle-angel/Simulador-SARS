const express = require('express');
const app = express();

app.set('trust proxy', true); 

app.get('/', (req, res) => {
    const ip = req.headers['x-forwarded-for'] || req.socket.remoteAddress;
    const userAgent = req.headers['user-agent'] || 'Desconocido';
    const language = req.headers['accept-language'] || 'Desconocido';
    const referrer = req.headers['referer'] || 'Desconocido';
    
    console.log('--- Nueva Visita ---');
    console.log('IP:', ip);
    console.log('User-Agent:', userAgent);
    console.log('Idioma:', language);
    console.log('Referrer:', referrer);
    
    res.set('Cache-Control', 'no-store');
    res.send('Página cargada');
});

app.listen(process.env.PORT || 3000, () => console.log('Servidor activo'));
