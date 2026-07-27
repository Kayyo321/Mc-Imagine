package com.mcimagine.ui;

import com.mcimagine.api.ModelInfo;
import com.mcimagine.model.ModelLoader;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.components.CycleButton;
import net.minecraft.client.gui.components.MultiLineEditBox;
import net.minecraft.client.gui.components.MultiLineTextWidget;
import net.minecraft.client.gui.components.StringWidget;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.gui.screens.worldselection.CreateWorldScreen;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.Component;

import java.nio.file.Path;
import java.util.List;

/**
 * Custom configuration screen for the Mc-Imagine AI world preset.
 * Opened when the user clicks "Customize" after selecting "Mc-Imagine AI World" as the world type.
 */
public class McImagineCustomizeScreen extends Screen {

    private final CreateWorldScreen parent;
    private MultiLineEditBox promptBox;
    private Button doneButton;
    private List<ModelInfo> availableModels;

    private DropdownWidget<ModelInfo> modelDropdown;

    public McImagineCustomizeScreen(CreateWorldScreen parent) {
        super(Component.translatable("mcimagine.gui.customize.title"));
        this.parent = parent;
    }

    @Override
    protected void init() {
        // --- Discover models ---
        Path modelsDir = Minecraft.getInstance().gameDirectory.toPath().resolve("mcimagine").resolve("models");
        ModelLoader loader = new ModelLoader(modelsDir);
        this.availableModels = loader.discoverModels();

        int centerX = this.width / 2;
        int contentWidth = 310;
        int leftEdge = centerX - contentWidth / 2;

        // --- Title ---
        this.addRenderableWidget(new StringWidget(
                leftEdge, 15, contentWidth, 20,
                this.title,
                this.font
        ));

        // --- Prompt label ---
        this.addRenderableWidget(new StringWidget(
                leftEdge, 40, contentWidth, 10,
                Component.translatable("mcimagine.gui.customize.prompt_label"),
                this.font
        ));

        // --- Prompt text box ---
        this.promptBox = new MultiLineEditBox(
                this.font,
                leftEdge, 55,
                contentWidth, 80,
                Component.translatable("mcimagine.gui.prompt"),
                Component.translatable("mcimagine.gui.prompt.hint")
        );
        // Restore any previously entered prompt
        if (!McImagineCreateWorldMode.currentPrompt.isEmpty()) {
            this.promptBox.setValue(McImagineCreateWorldMode.currentPrompt);
        }
        this.addRenderableWidget(this.promptBox);

        // --- Done button ---
        this.doneButton = Button.builder(
                Component.translatable("mcimagine.gui.customize.done"),
                (btn) -> {
                    // Save prompt to shared state
                    McImagineCreateWorldMode.currentPrompt = this.promptBox.getValue();
                    this.minecraft.setScreen(this.parent);
                }
        ).bounds(centerX - 155, this.height - 28, 150, 20).build();
        // Disable Done if no models are available
        this.doneButton.active = !this.availableModels.isEmpty();
        this.addRenderableWidget(this.doneButton);

        // --- Cancel button ---
        this.addRenderableWidget(Button.builder(
                Component.translatable("mcimagine.gui.customize.cancel"),
                (btn) -> this.minecraft.setScreen(this.parent)
        ).bounds(centerX + 5, this.height - 28, 150, 20).build());

        // --- Model selection or error message ---
        // Added LAST so it renders on top and receives clicks first
        if (this.availableModels.isEmpty()) {
            // No models found — show error message
            String errorText = Component.translatable(
                    "mcimagine.gui.customize.no_models",
                    modelsDir.toAbsolutePath().toString()
            ).getString();

            MultiLineTextWidget errorWidget = new MultiLineTextWidget(
                    Component.literal(errorText),
                    this.font
            ).setMaxWidth(contentWidth);
            errorWidget.setX(leftEdge);
            errorWidget.setY(145);
            this.addRenderableWidget(errorWidget);
        } else {
            // Resolve the currently selected model, defaulting to the first one
            ModelInfo initialModel = this.availableModels.stream()
                    .filter(m -> m.path().equals(McImagineCreateWorldMode.selectedModelPath))
                    .findFirst()
                    .orElse(this.availableModels.get(0));

            // Initialize state if not yet set
            if (McImagineCreateWorldMode.selectedModelPath.isEmpty()) {
                McImagineCreateWorldMode.selectedModelPath = initialModel.path();
                McImagineCreateWorldMode.selectedModelName = initialModel.name();
            }

            this.modelDropdown = new DropdownWidget<>(
                    leftEdge, 145, contentWidth, 20,
                    Component.translatable("mcimagine.gui.model"),
                    this.availableModels,
                    initialModel,
                    model -> Component.literal(model.name()),
                    value -> {
                        McImagineCreateWorldMode.selectedModelPath = value.path();
                        McImagineCreateWorldMode.selectedModelName = value.name();
                    }
            );
            this.addRenderableWidget(this.modelDropdown);
        }
    }

    @Override
    public void render(GuiGraphics guiGraphics, int mouseX, int mouseY, float partialTick) {
        this.renderBackground(guiGraphics);
        super.render(guiGraphics, mouseX, mouseY, partialTick);
        
        // Render the dropdown list last so it appears over other elements
        if (this.modelDropdown != null) {
            this.modelDropdown.renderDropdown(guiGraphics, mouseX, mouseY, partialTick);
        }
    }

    @Override
    public void onClose() {
        this.minecraft.setScreen(this.parent);
    }
}
