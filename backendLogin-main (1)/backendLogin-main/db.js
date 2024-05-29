const mongoose = require("mongoose");

module.exports = () => {
	const connectionParams = {
		useNewUrlParser: true,
		useUnifiedTopology: true,
	};
	try {
		mongoose.connect("mongodb+srv://noob:noob@cluster0.o7cphtt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0").then(()=>{
            console.log("mongodb is connected");
      })
	} catch (error) {
		console.log(error);
		console.log("Could not connect database!");
	}
};
