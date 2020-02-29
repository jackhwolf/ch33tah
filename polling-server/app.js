const express = require("express");
const http = require("http");
const socketIo = require("socket.io");
const fetch = require("node-fetch")

const port = process.env.PORT || 4001;
const app = express();
const server = http.createServer(app);
const io = socketIo(server); 
const router = express.Router();

const getApiAndEmit = async socket => {
    const data = await fetch('http://127.0.0.1:5000/v1/time')
        .then((response) => {
            return response.json();
        })
    socket.emit("data", data.time);
};

io.on("connection", socket => {
    console.log("New client connected"), setInterval(
        () => getApiAndEmit(socket),
        2000
    );
    socket.on("disconnect", () => console.log("Client disconnected"));
});


server.listen(port, () => console.log(`Listening on port ${port}`));