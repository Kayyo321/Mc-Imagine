package com.mcimagine.ui;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.gui.components.Button;
import net.minecraft.network.chat.Component;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;

public class DropdownWidget<T> extends Button {
    private final List<T> options;
    private T selected;
    private boolean expanded = false;
    private int scrollOffset = 0;
    private final Consumer<T> onSelect;
    private final Function<T, Component> formatter;
    private final Component prefix;
    private final int maxOptionsVisible = 5;

    public DropdownWidget(int x, int y, int width, int height, Component prefix, List<T> options, T initial, Function<T, Component> formatter, Consumer<T> onSelect) {
        super(x, y, width, height, Component.empty(), btn -> {}, Button.DEFAULT_NARRATION);
        this.prefix = prefix;
        this.options = options;
        this.selected = initial;
        this.formatter = formatter;
        this.onSelect = onSelect;
        this.updateMessage();
    }

    private void updateMessage() {
        Component text = Component.empty().append(prefix).append(": ").append(formatter.apply(selected)).append(expanded ? " ▲" : " ▼");
        this.setMessage(text);
    }

    @Override
    public void onPress() {
        expanded = !expanded;
        updateMessage();
    }

    @Override
    public boolean mouseScrolled(double mouseX, double mouseY, double delta) {
        if (expanded && options.size() > maxOptionsVisible) {
            int maxScroll = options.size() - maxOptionsVisible;
            if (delta > 0 && scrollOffset > 0) {
                scrollOffset--;
                return true;
            } else if (delta < 0 && scrollOffset < maxScroll) {
                scrollOffset++;
                return true;
            }
        }
        return super.mouseScrolled(mouseX, mouseY, delta);
    }

    @Override
    public boolean mouseClicked(double mouseX, double mouseY, int button) {
        if (this.active && this.visible) {
            if (expanded) {
                int listY = this.getY() + this.getHeight();
                int listHeight = Math.min(options.size(), maxOptionsVisible) * this.getHeight();
                
                // If clicked inside the dropdown list
                if (mouseX >= this.getX() && mouseX < this.getX() + this.getWidth() && mouseY >= listY && mouseY < listY + listHeight) {
                    int clickedIndex = (int) ((mouseY - listY) / this.getHeight());
                    int actualIndex = clickedIndex + scrollOffset;
                    if (actualIndex >= 0 && actualIndex < options.size()) {
                        selected = options.get(actualIndex);
                        onSelect.accept(selected);
                        this.playDownSound(Minecraft.getInstance().getSoundManager());
                        expanded = false;
                        updateMessage();
                        return true;
                    }
                } else if (!this.isMouseOver(mouseX, mouseY)) {
                    // Clicked outside both the button and the list
                    expanded = false;
                    updateMessage();
                    // Don't consume the click if it's outside
                    return false;
                }
            }
        }
        return super.mouseClicked(mouseX, mouseY, button);
    }

    @Override
    public boolean isMouseOver(double mouseX, double mouseY) {
        if (super.isMouseOver(mouseX, mouseY)) {
            return true;
        }
        if (expanded) {
            int listY = this.getY() + this.getHeight();
            int listHeight = Math.min(options.size(), maxOptionsVisible) * this.getHeight();
            return mouseX >= this.getX() && mouseX < this.getX() + this.getWidth() && mouseY >= listY && mouseY < listY + listHeight;
        }
        return false;
    }

    // Called by the parent screen to draw the dropdown list over other elements
    public void renderDropdown(GuiGraphics graphics, int mouseX, int mouseY, float partialTick) {
        if (expanded && this.visible) {
            graphics.pose().pushPose();
            graphics.pose().translate(0, 0, 400); // Bring to front to prevent text bleeding from underneath

            int listY = this.getY() + this.getHeight();
            int listHeight = Math.min(options.size(), maxOptionsVisible) * this.getHeight();
            
            // Draw background
            graphics.fill(this.getX(), listY, this.getX() + this.getWidth(), listY + listHeight, 0xFF000000);
            
            // Draw border
            graphics.renderOutline(this.getX(), listY, this.getWidth(), listHeight, 0xFFFFFFFF);

            Minecraft minecraft = Minecraft.getInstance();
            for (int i = 0; i < maxOptionsVisible; i++) {
                int actualIndex = i + scrollOffset;
                if (actualIndex >= options.size()) break;
                
                T opt = options.get(actualIndex);
                int optY = listY + i * this.getHeight();
                boolean hovered = mouseX >= this.getX() && mouseX < this.getX() + this.getWidth() && mouseY >= optY && mouseY < optY + this.getHeight();
                
                if (hovered) {
                    graphics.fill(this.getX() + 1, optY + 1, this.getX() + this.getWidth() - 1, optY + this.getHeight() - 1, 0xFF555555);
                }
                
                // Render scrolling string to handle long names
                renderScrollingString(graphics, minecraft.font, formatter.apply(opt), this.getX() + 5, optY, this.getX() + this.getWidth() - 10, optY + this.getHeight(), 16777215);
            }
            
            // Draw scrollbar indicator if scrollable
            if (options.size() > maxOptionsVisible) {
                int maxScroll = options.size() - maxOptionsVisible;
                int scrollbarHeight = Math.max(10, listHeight * maxOptionsVisible / options.size());
                int scrollbarY = listY + (listHeight - scrollbarHeight) * scrollOffset / maxScroll;
                graphics.fill(this.getX() + this.getWidth() - 4, scrollbarY, this.getX() + this.getWidth() - 1, scrollbarY + scrollbarHeight, 0xFFAAAAAA);
            }

            graphics.pose().popPose();
        }
    }
}
