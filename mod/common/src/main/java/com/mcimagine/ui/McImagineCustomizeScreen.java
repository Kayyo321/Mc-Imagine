package com.mcimagine.ui;

import com.mcimagine.api.ModelInfo;
import com.mcimagine.generation.ImagineBiomeSource;
import com.mcimagine.generation.ImagineChunkGenerator;
import com.mcimagine.model.BiomePalette;
import com.mcimagine.model.ModelLoader;
import com.mcimagine.prompt.ParsedPrompt;
import com.mcimagine.prompt.PromptParser;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.gui.components.Button;
import net.minecraft.client.gui.components.MultiLineEditBox;
import net.minecraft.client.gui.components.MultiLineTextWidget;
import net.minecraft.client.gui.components.StringWidget;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.gui.screens.worldselection.CreateWorldScreen;
import net.minecraft.client.gui.screens.worldselection.WorldCreationContext;
import net.minecraft.core.Holder;
import net.minecraft.core.Registry;
import net.minecraft.core.RegistryAccess;
import net.minecraft.core.registries.Registries;
import net.minecraft.network.chat.Component;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.biome.Biomes;
import net.minecraft.world.level.biome.FixedBiomeSource;
import net.minecraft.world.level.levelgen.NoiseGeneratorSettings;
import net.minecraft.world.level.levelgen.WorldOptions;

import java.nio.file.Path;
import java.util.List;

/**
 * Custom configuration screen for the Mc-Imagine AI world preset.
 * Opened when the user clicks "Customize" after selecting "Mc-Imagine AI World" as the world type.
 */
public class McImagineCustomizeScreen extends Screen {

    private final CreateWorldScreen parent;
    private final WorldCreationContext creationContext;
    private MultiLineEditBox promptBox;
    private MultiLineTextWidget warningsWidget;
    private Button doneButton;
    private List<ModelInfo> availableModels;
    private ModelInfo selectedModel;

    private DropdownWidget<ModelInfo> modelDropdown;

    public McImagineCustomizeScreen(CreateWorldScreen parent, WorldCreationContext creationContext) {
        super(Component.translatable("mcimagine.gui.customize.title"));
        this.parent = parent;
        this.creationContext = creationContext;
    }

    @Override
    protected void init() {
        // --- Discover models ---
        Path modelsDir = Minecraft.getInstance().gameDirectory.toPath().resolve("mcimagine").resolve("models");
        ModelLoader loader = new ModelLoader(modelsDir);
        // Only models that passed their load-time trial inference are offered - an unavailable model would
        // otherwise fail on the very first chunk request instead of here (PROJECT.md §"Error Handling &
        // Fallback Chain").
        this.availableModels = loader.discoverModels().stream().filter(ModelInfo::available).toList();

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
                    McImagineCreateWorldMode.currentPrompt = this.promptBox.getValue();
                    applySelectionToWorld();
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

        // --- Warnings widget (M7 prompt-tag validation, shown before Done is used) ---
        this.warningsWidget = new MultiLineTextWidget(Component.empty(), this.font).setMaxWidth(contentWidth);
        this.warningsWidget.setX(leftEdge);
        this.warningsWidget.setY(170);
        this.addRenderableWidget(this.warningsWidget);

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
            this.selectedModel = initialModel;

            this.modelDropdown = new DropdownWidget<>(
                    leftEdge, 145, contentWidth, 20,
                    Component.translatable("mcimagine.gui.model"),
                    this.availableModels,
                    initialModel,
                    model -> Component.literal(model.name()),
                    value -> {
                        McImagineCreateWorldMode.selectedModelPath = value.path();
                        McImagineCreateWorldMode.selectedModelName = value.name();
                        this.selectedModel = value;
                    }
            );
            this.addRenderableWidget(this.modelDropdown);
        }

        updateWarnings();
    }

    /** Re-tags the current prompt against the selected model's capabilities and refreshes the warnings text. */
    private void updateWarnings() {
        if (warningsWidget == null) {
            return;
        }
        if (selectedModel == null || promptBox == null) {
            warningsWidget.setMessage(Component.empty());
            return;
        }
        ParsedPrompt parsed = new PromptParser().parse(promptBox.getValue(), selectedModel.capabilities());
        if (parsed.warnings().isEmpty()) {
            warningsWidget.setMessage(Component.empty());
        } else {
            String joined = String.join("\n", parsed.warnings());
            warningsWidget.setMessage(Component.literal(joined));
        }
    }

    /**
     * Builds a new {@link ImagineChunkGenerator} carrying the prompt/model/seed and installs it into the
     * overworld {@code LevelStem} via {@code WorldCreationUiState.updateDimensions} /
     * {@code WorldDimensions.replaceOverworldGenerator} - this is what makes the prompt+model actually reach
     * generation (docs/poc-plan.md Phase 2 - previously the prompt was written to a static field and never
     * read again).
     */
    private void applySelectionToWorld() {
        if (creationContext == null || selectedModel == null) {
            return;
        }

        String prompt = promptBox.getValue();
        String modelId = selectedModel.id();

        RegistryAccess.Frozen registryAccess = creationContext.worldgenLoadContext();
        Registry<Biome> biomeRegistry = registryAccess.registryOrThrow(Registries.BIOME);
        Registry<NoiseGeneratorSettings> noiseSettingsRegistry = registryAccess.registryOrThrow(Registries.NOISE_SETTINGS);
        Holder<NoiseGeneratorSettings> settings = noiseSettingsRegistry.getHolderOrThrow(NoiseGeneratorSettings.OVERWORLD);
        Holder<Biome> plains = biomeRegistry.getHolderOrThrow(Biomes.PLAINS);

        List<String> biomePaletteIds = selectedModel.capabilities().biomePalette();
        BiomeSource biomeSource;
        if (!biomePaletteIds.isEmpty()) {
            BiomePalette resolved = BiomePalette.resolveFrom(biomePaletteIds, biomeRegistry);
            biomeSource = new ImagineBiomeSource(resolved.palette(), plains);
        } else {
            // docs/model-spec.md's Versioning note: a manifest omitting biome_palette means biome
            // assignment stays vanilla-default rather than being driven by biome_grid at all.
            biomeSource = new FixedBiomeSource(plains);
        }

        long seed = WorldOptions.parseSeed(parent.getUiState().getSeed()).orElseGet(WorldOptions::randomSeed);

        ImagineChunkGenerator newGenerator = new ImagineChunkGenerator(biomeSource, settings, prompt, modelId, seed);

        parent.getUiState().updateDimensions((regAccess, worldDimensions) ->
                worldDimensions.replaceOverworldGenerator(regAccess, newGenerator));
    }

    @Override
    public void tick() {
        super.tick();
        updateWarnings();
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
