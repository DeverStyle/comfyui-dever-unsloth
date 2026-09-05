import { app } from "/scripts/app.js";

app.registerExtension({
  name: "Comfy.UnslothNode",
  aboutPageBadges: [
    {
      label: "comfyui-dever-unsloth",
      url: "https://github.com/dever/comfyui-dever-unsloth",
      icon: "pi pi-github",
    },
  ],
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "UnslothConnectivity") {
      return;
    }

    const originalNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function () {
      if (originalNodeCreated) {
        originalNodeCreated.apply(this, arguments);
      }

      const urlWidget = this.widgets.find((w) => w.name === "url");
      const keyWidget = this.widgets.find((w) => w.name === "api_key");
      const modelWidget = this.widgets.find((w) => w.name === "model");
      const refreshButtonWidget = this.addWidget("button", "🔄 Reconnect");

      const fetchModels = async (url, api_key) => {
        const response = await fetch("/unsloth/get_models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url, api_key }),
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || response.statusText);
        }
        return data;
      };

      const updateModels = async () => {
        refreshButtonWidget.name = "⏳ Fetching...";
        const url = urlWidget.value;
        const api_key = keyWidget.value;

        let models = [];
        try {
          models = await fetchModels(url, api_key);
        } catch (error) {
          console.error("Error fetching models:", error);
          app.extensionManager.toast.add({
            severity: "error",
            summary: "Unsloth connection error",
            detail:
              "Make sure Unsloth Desktop is running and the API key is correct.",
            life: 5000,
          });
          refreshButtonWidget.name = "🔄 Reconnect";
          this.setDirtyCanvas(true);
          return;
        }

        const prevValue = modelWidget.value;

        modelWidget.options.values = models;

        if (models.includes(prevValue)) {
          modelWidget.value = prevValue; // stay on current.
        } else if (models.length > 0) {
          modelWidget.value = models[0]; // set first as default.
        }

        refreshButtonWidget.name = "🔄 Reconnect";
        this.setDirtyCanvas(true);
      };

      urlWidget.callback = updateModels;
      keyWidget.callback = updateModels;
      refreshButtonWidget.callback = updateModels;

      const dummy = async () => {
        // Calling an async method ensures the widgets hold their actual values
        // from the page, not the defaults from the node definition.
      };

      // Initial update.
      await dummy();
      await updateModels();
    };
  },
});
