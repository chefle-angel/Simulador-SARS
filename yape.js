const express = require('express');
const app = express();

// Necesario en Render para leer la IP real detrás de su proxy
app.set('trust proxy', true); 

app.get('/', (req, res) => {
    const ip = req.headers['x-forwarded-for'] || req.socket.remoteAddress;
    console.log('IP capturada:', ip);
    res.send('Página cargada');
});

app.listen(process.env.PORT || 3000, () => console.log('Servidor activo'));
