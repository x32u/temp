module.exports = {
    apps: [
      {
        name: 'api',
        script: 'python3.11',
        args: 'main.py',
        interpreter: 'none', // This tells PM2 not to use Node.js to run the script
      },
    ],
  };
